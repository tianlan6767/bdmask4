"""
python export_model_to_onnx-dy.py --weights model_0830a.pth --output adet.onnx
/media/ps/train/HJL/Trt_inference/trt852cuda115cudnn8/bin/trtexec --onnx=1218.onnx --saveEngine=1218 --minShapes=input_image:1x3x2048x2048 --optShapes=input_image:1x3x2048x2048 --maxShapes=input_image:1x3x2048x2048 --fp16 --device=0 --workspace=10240
/media/ps/train/HJL/Trt_inference/trt852cuda115cudnn8/bin/trtexec --onnx=1218.onnx --saveEngine=1218 --minShapes=input_image:1x3x2048x2048 --optShapes=input_image:1x3x4096x4096 --maxShapes=input_image:1x3x4096x4096 --fp16 --device=0 --workspace=40960
"""


import os
import json
import glob
import time
import cv2
import torch
import math
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
from typing import List,Tuple
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.ops import roi_align
from torchvision.ops import boxes as box_ops


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)
  
    
def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)


def _fmt_box_list(box_tensor, batch_index: int):
    repeated_index = torch.full_like(
        box_tensor[:, :1], batch_index, dtype=box_tensor.dtype, device=box_tensor.device
    )
    return cat((repeated_index, box_tensor), dim=1)


def assign_boxes_to_levels(
    box_lists: List,
    min_level: int,
    max_level: int,
    canonical_box_size: int,
    canonical_level: int,
):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]): A list of N Boxes or N RotatedBoxes,
            where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels (sqrt(box area)).
        canonical_level (int): The feature map level index on which a canonically-sized box
            should be placed.

    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
      
      
def paste_masks_in_image(
    masks: torch.Tensor, boxes, image_shape: Tuple[int, int], threshold: float = 0.5
):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.

    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """

    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu" or torch.jit.is_scripting():
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            # for visualization and debugging
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        if torch.jit.is_scripting():  # Scripting does not use the optimized codepath
            img_masks[inds] = masks_chunk
        else:
            img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks
  
  
class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.

        Note:
            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
            roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect alignment
            (relative to our pixel model) when performing bilinear interpolation.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors; see
            detectron2/tests/test_roi_align.py for verification.

            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
        """
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        return roi_align(
            input,
            rois.to(dtype=input.dtype),
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
    ):
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        if pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def forward(self, x: List[torch.Tensor], box_lists: List):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
            len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )
        if len(box_lists) == 0:
            return torch.zeros(
                (0, x[0].shape[1]) + self.output_size, device=x[0].device, dtype=x[0].dtype
            )

        pooler_fmt_boxes = cat([_fmt_box_list(box_list, i) for i, box_list in enumerate(box_lists)], dim=0)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        num_boxes = pooler_fmt_boxes.size(0)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )

        for level, pooler in enumerate(self.level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x[level], pooler_fmt_boxes_level)

        return output

############################# self  ################################

def make_dir(dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
  
def commom_compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )       # [0,8,16,24,...,2040]
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )       # [0,8,16,24,...,2040]
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations
    
    
def fcos_compute_locations(strides,features,device="cpu"):
    """
    # [torch.Size([65536, 2]), torch.Size([16384, 2]), torch.Size([4096, 2]), torch.Size([1024, 2]), torch.Size([256, 2])]
    """
    locations = []
    for level, feature in enumerate(features):
        h, w = feature
        locations_per_level = commom_compute_locations(h, w, strides[level],device)
        locations.append(locations_per_level)
    return torch.cat(locations,dim=0)


class Blender:
    def __init__(self, cfg):
        self.pooler_resolution = cfg["BLENDMASK"]["BOTTOM_RESOLUTION"]
        sampling_ratio         = cfg["BLENDMASK"]["POOLER_SAMPLING_RATIO"]
        pooler_type            = cfg["BLENDMASK"]["POOLER_TYPE"]
        pooler_scales          = cfg["BLENDMASK"]["POOLER_SCALES"]
        self.attn_size         = cfg["BLENDMASK"]["ATTN_SIZE"]
        self.top_interp        = cfg["BLENDMASK"]["TOP_INTERP"]
        num_bases              = cfg["BASIS_MODULE"]["NUM_BASES"]
        self.attn_len = num_bases * self.attn_size * self.attn_size
        self.pooler = ROIPooler(
            output_size=self.pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_level=2)
    
    def __call__(self, bases, pred_boxes,top_feat):
        total_instances = pred_boxes.shape[0]
        if total_instances == 0:
            return []
        rois = self.pooler(bases, [torch.as_tensor(pred_boxes, dtype=torch.float32)])
        attns = top_feat
        pred_mask_logits = self.merge_bases(rois, attns).sigmoid()
        pred_mask_logits = pred_mask_logits.view(
            -1, 1, self.pooler_resolution, self.pooler_resolution)
        return pred_mask_logits
    
    def merge_bases(self, rois, coeffs, location_to_inds=None):
        # merge predictions
        N = coeffs.size(0)
        if location_to_inds is not None:
            rois = rois[location_to_inds]
        N, B, H, W = rois.size()
        coeffs = coeffs.view(N, -1, self.attn_size, self.attn_size)
        coeffs = F.interpolate(coeffs, (H, W),
                               mode=self.top_interp, align_corners=False).softmax(dim=1)
        masks_preds = (rois * coeffs).sum(dim=1)
        return masks_preds.view(N, -1)


class VisualDemo:
    def __init__(self,dst,save_json=False,save_img=True):
        self.color = (0, 255, 0)
        self.thickness = 2
        self.font_size = 1
        self.dst = dst
        self.save_json = save_json
        self.save_img = save_img
        self.new_json = {}
    
    def draw_polygon(self,im, pts):
        cv2.polylines(im, pts=pts, isClosed=True,color=self.color, thickness=self.thickness)
        
    def draw_rect(self,im, bbox):
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2],bbox[3]), color=self.color, thickness=self.thickness)

    def draw_text(self,im, value, x,y):
        cv2.putText(im, value, (x,y),cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.color, self.thickness)
        
    def __call__(self,filename,im,bboxes,labels,scores,bitmasks):
        self.new_json[filename] = {}
        self.new_json[filename]["filename"] = filename
        regions_list = []
        for i in range(bitmasks.shape[0]):
            contours, _ = cv2.findContours(bitmasks[i].numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                polygons = contour.flatten().tolist()
                xs, ys = polygons[0::2], polygons[1::2]
            score = str(round(np.sqrt(scores[i].numpy()),2))
            label = str(labels[i].item()+1)
            if self.save_img:
                self.draw_polygon(im,[np.dstack((xs,ys))])
                self.draw_text(im,"{} s:{}".format(label,score), int(bboxes[i][0].item()),int(bboxes[i][1].item()))
            if self.save_json:
                new_dict = {'shape_attributes':{'all_points_x':xs,'all_points_y':ys},
                            'region_attributes':{'regions':label,"score":score}}
                regions_list.append(new_dict)
        self.new_json[filename]["regions"] = regions_list
        
        if self.save_img:
            cv2.imwrite(os.path.join(self.dst,filename),im)
        if self.save_json:
            with open(os.path.join(self.dst,"trt.json"), "w") as f:
                json.dump(self.new_json, f)


class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

  
class AdetTRT:
    def __init__(self, engine_file_path,device=0):
        # Create a Context on this device,
        # self.ctx = cuda.Device(device).make_context()
        stream  = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        trt.init_libnvinfer_plugins(None, "")

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine= runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        context.active_optimization_profile = 0
        
        self.args = {"input_image":0,"bases":1,"pred":2}
        self.batch_size = engine.max_batch_size
        self.engine = engine
        self.context = context
        self.stream = stream
        
    def allocate_buffer(self,num_locations,image_shape):
        h,w = image_shape
        inputs = []
        outputs = []
        bindings = []
        # 分配动态内存空间
        for binding in self.engine:
            if binding == "input_image":
                self.context.set_binding_shape(self.args[binding],tuple([1,3,h,w]))
                print("error0")
            elif binding == "bases":
                self.context.set_binding_shape(self.args[binding],(1,4,h//4,w//4))
                print("error1")
            else:
                self.context.set_binding_shape(self.args[binding],(1,num_locations,814))
                print("error2")
            size = trt.volume(self.context.get_binding_shape(self.args[binding])) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)       
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem,device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem,device_mem))
        return inputs,outputs,bindings
        
    def infer(self,batch_input_image,num_locations):
        inputs,outputs,bindings = self.allocate_buffer(num_locations,batch_input_image.shape[2:])
        # start = time.time()
        [cuda.memcpy_htod_async(inp.device, batch_input_image.ravel(), self.stream) for inp in inputs]
        self.context.execute_async(batch_size=self.batch_size,bindings=bindings, stream_handle=self.stream.handle)   
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in outputs]
        # print("cost:",time.time()-start)
        self.stream.synchronize()
        return [out.host for out in outputs]
      
        
class ImageProcss:
    def __init__(self,path,mean,std):
        self.pixel_mean = np.array(mean).reshape(3,1,1)
        self.pixel_std = np.array(std).reshape(3,1,1)
        self.imgs = glob.glob(path+'/*.[jb][pm][gp]')*100
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std
  
    def pre(self,im):
        image = self.normalizer(im.transpose((2,0,1)))
        image = image.astype('float32')
        return image[None,:]
      
    def __getitem__(self,index):
        im = cv2.imread(self.imgs[index],1)
        return os.path.basename(self.imgs[index]),im,self.pre(im)
    
    def __len__(self):
        return len(self.imgs)
    

def main():
    start0 = time.time()
    cfg = {
        "FCOS":{
          "INFERENCE_TH_TEST":0.25
          },
        "BLENDMASK":{
          "BOTTOM_RESOLUTION":56,
          "POOLER_SAMPLING_RATIO":1,
          "POOLER_TYPE":"ROIAlignV2",
          "POOLER_SCALES":[0.25],
          "ATTN_SIZE":14,
          "TOP_INTERP":"bilinear",
          },
        "BASIS_MODULE":{
          "NUM_BASES":4,
          }
        }
    MEAN = [47,47,49]
    STD= [24,24,25]
    SCORE_TH = 0.25
    NMS_TH = 0.1
    STRIDES = [8, 16, 32, 64, 128]
    ALL_LOCATIONS = {
      "2048,2048": fcos_compute_locations(STRIDES,[(2048//stride,2048//stride) for stride in STRIDES]),
      "4096,4096": fcos_compute_locations(STRIDES,[(4096//stride,4096//stride) for stride in STRIDES]),
      "3072,4096": fcos_compute_locations(STRIDES,[(3072//stride,4096//stride) for stride in STRIDES]),
    }
    # trt_path = r"/media/ps/244e88e1-d2e1-477f-9e37-7b9cb43b842a/LB/train/trt/static.trt"     # 固定尺寸
    trt_path = r"/media/ps/244e88e1-d2e1-477f-9e37-7b9cb43b842a/LB/train/trt/dy.trt"  # 动态尺寸
    image_path = r"/media/ps/244e88e1-d2e1-477f-9e37-7b9cb43b842a/LB/train/trt/images"
    dst = image_path+"_dy_trt"
    make_dir(dst)

    ip = ImageProcss(image_path,MEAN,STD)
    predictor = AdetTRT(trt_path)
    blender = Blender(cfg)
    vd = VisualDemo(dst,save_img=False,save_json=False)
    
    # t0 = time.time()
    for n in tqdm(range(len(ip))):
        filename,ori_image,image = ip[n]
        h,w = ori_image.shape[:2]
        
        if str(h)+","+str(w) not in ALL_LOCATIONS.keys():
            locations = fcos_compute_locations(STRIDES,[(h//stride,w//stride) for stride in STRIDES])
        else:
            locations = ALL_LOCATIONS[str(h)+","+str(w)]
        
        # 推理
        trt_base,trt_pred = predictor.infer(image,locations.shape[0])
        trt_base = trt_base.reshape((1,4,512*(h//2048),512*(w//2048)))
        trt_pred = trt_pred.reshape((1,locations.shape[0],814))
        
        # 筛选pred得分大于阈值，切分出bbox,score,labels
        _,keep = np.where(trt_pred[...,4]>SCORE_TH)
        reg_pred,ctrness_pred,logits_pred,top_feats = np.split(trt_pred[:,keep,:],[4,5,30],axis=2)

        # 将bbox覆盖到符合阈值的location上
        keep_locations = locations[keep,:]
        x1 = keep_locations[...,0,None] - reg_pred[...,0,None]
        y1 = keep_locations[...,1,None] - reg_pred[...,1,None]
        x2 = keep_locations[...,0,None] + reg_pred[...,0,None]
        y2 = keep_locations[...,1,None] + reg_pred[...,1,None]
        bboxes = torch.cat([x1,y1,x2,y2],dim=2).squeeze()
        scores = torch.tensor(ctrness_pred.flatten())
        labels = torch.tensor(np.argmax(logits_pred,axis=2).squeeze())
        
        # NMS
        keep_nms =  box_ops.batched_nms(bboxes, scores, labels, NMS_TH)
        bboxes = bboxes[keep_nms]
        scores = scores[keep_nms]
        labels = labels[keep_nms]
        top_feats = top_feats[:,keep_nms,:]
        
        # 计算MASK
        proposals = blender([torch.from_numpy(trt_base)],bboxes,torch.from_numpy(top_feats).squeeze())
        bitmasks = paste_masks_in_image(proposals[:,0,:,:],bboxes,(h,w),threshold=0.5)
        
        # vd(filename,ori_image,bboxes,labels,scores,bitmasks)
        
       # break
        

    print("cost:",time.time()-start0)

if __name__ == "__main__":
    main()
