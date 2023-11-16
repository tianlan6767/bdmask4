import onnx
from onnx.tools import update_model_dims


def get_module_onnx_info():
    """获取onnx库的版本和算⼦库版本信息"""
    print(f"onnx.version={onnx.__version__}, opset={onnx.defs.onnx_opset_version()} "
    "IR_VERSION={onnx.IR_VERSION} .")

def get_model_info(model):
    """打印model的meta info"""
    for field in ['doc_string', 'domain', 'functions',
    'ir_version', 'metadata_props', 'model_version',
    'opset_import', 'producer_name', 'producer_version',
    'training_info']:
        print(f"{field}: {getattr(model, field)}")


def load_model(onnx_path):
    """加载onnx 模型⽂件"""
    model = onnx.load(onnx_path)
    return model

def check_model(model):
    """检查onnx 模型的有效性"""
    try:
    onnx.checker.check_model(model)
    except Exception as e:
    print(e)

def print_model_info(model):
    """
    查看模型的输入，输出和节点信息
    """
    print('** inputs **')
    print(model.graph.input)
    # the list of outputs
    print('** outputs **')
    print(model.graph.output)
    # the list of nodes
    print('** nodes **')
    print(model.graph.node)
    
    
def shape2tuple(shape):
    """打印维度时⽤0替代动态维度"""
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

 
def pretty_print_model_info(model):
    """以更好看的⽅式输出模型的输⼊、输出和节点信息"""
    print('** inputs **')
    for obj in model.graph.input:
        print("name=%r dtype=%r shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))
        print('** outputs **')
    for obj in model.graph.output:
        print("name=%r dtype=%r shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))
        print('** nodes **')
    for node in model.graph.node:
        print("name=%r type=%r input=%r output=%r" % (
        node.name, node.op_type, node.input, node.output))

def serialize_to_onnx(model, save_file_path):
    """模型序列化."""
    with open(save_file_path, "wb") as fp:
    fp.write(model.SerializeToString())
def save_onnx_file(model, save_file_path):
    """保存模型"""
    onnx.save_model(model, save_file_path)
def shape_inference_model(model):
    """在张量类型和形状确定的情况下，运⾏时可以预先估计内存消耗并优化计算"""
    model = onnx.shape_inference.infer_shapes(model)
    return model

def evaluator_model(model, input, verbose=1):
    """验证结果, 不考虑性能和优化"""
    sess = onnx.reference.ReferenceEvaluator(model)
    print(sess.run(None, input))


def change_version(model, target_version):
    """转变opset版本号"""
    converted_model = onnx.version_converter.convert_version(model,
    target_version)
    return converted_model
def extract_sub_model(input_path, output_path, start_node, end_node):
    """分割onnx⽂件, 提取⼦模型, 合并参看ONNX Compose
    eg:
    input_path = "path/to/the/original/model.onnx"
    output_path = "path/to/save/the/extracted/model.onnx"
    input_names = ["input_0", "input_1", "input_2"]
    output_names = ["output_0", "output_1"]
    onnx.utils.extract_model(input_path, output_path, input_names,
    output_names)
    """
    onnx.utils.extract_model(input_path, output_path, start_node, end_node)
def update_model_inputs_outputs_dims(model, input_dict, output_dict):
    """修改输⼊:
    ⽀持某个维度从动态和静态相互转换, 但是修改静态到静态不⽀持
    eg:
    input_dict = {
    "input": [1, 1024],
    "h0": [2, 1, 64],
    "c0": [2, 1, 64]
    }
    output_dict = {
    "output": [1, 2, 1],
    "hn": [2, 1, 64],
    "cn": [2, 1, 64],
    }
    """
    update_model_dims.update_inputs_outputs_dims(model, input_dict,
        output_dict)

def simplify_model(model):
    """利⽤onnxsim来简化模型"""
    import onnxsim
    model_sim, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    return model_sim


model.opset_import[0].version = 11
def modify_Unsqueeze_to_11(model):
    """Unsqueeze 算⼦降级"""
    graph = model.graph
    node = graph.node
    unsqueeze_nodes = []
    for i in range(len(node)):
        if node[i].op_type == "Unsqueeze":
        unsqueeze_nodes.append(i)
        # print(unsqueeze_nodes)
    for idx in unsqueeze_nodes:
        unsqueeze_node = node[idx]
        # print(unsqueeze_node)
        input_name = unsqueeze_node.input[0]
        output_name = unsqueeze_node.output[0]
        node_name = unsqueeze_node.name
        node_type = unsqueeze_node.op_type
        new_node = onnx.helper.make_node(
        node_type,
        inputs=[input_name],
        outputs=[output_name],
        name=node_name,
        axes=[1],
        )
    node.remove(unsqueeze_node)
    node.insert(idx, new_node)                                          


def modify_Squeeze_to_11(model):
    """squeeze 算⼦降级"""
    graph = model.graph
    node = graph.node
    squeeze_nodes = []
    for i in range(len(node)):
    if node[i].op_type == "Squeeze":
    squeeze_nodes.append(i)
    # print(squeeze_nodes)
    for idx in squeeze_nodes:
    old_node = node[idx]
    # print(old_node)
    input_name = old_node.input[0]
    output_name = old_node.output[0]
    node_name = old_node.name
    node_type = old_node.op_type
    new_node = onnx.helper.make_node(
    node_type,
    inputs=[input_name],
    outputs=[output_name],
    name=node_name,
    axes=[1],
    )
    node.remove(old_node)
    node.insert(idx, new_node)


def export_model_from_pytorch_to_onnx(pytorch_model, onnx_model_name):
    batch_size = 1
    # input to the model
    x = torch.randn(batch_size, 1, 32, 32)
    out = pytorch_model(x)
    #print("out:", out)
    # export the model
    torch.onnx.export(pytorch_model, # model being run
        x, # model input (or a tuple for m
        onnx_model_name, # where to save the model (can
        export_params=True, # store the trained parameter w
        opset_version=9, # the ONNX version to export th
        do_constant_folding=True, # whether to execute constant f
        input_names = ['input'], # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={ # variable length axes
        'input' : {0 : 'batch_size'},
        'output': {0 : 'batch_size'}})


if __name__ == "__main__":
    onnx_path = r"/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/model_0364999-dy.onnx"
    model = onnx.load(onnx_path)
    get_model_info(model)
    pretty_print_model_info(model)



    origin_file_path = "/home/wz/document/Art.Studio-dev-0814/onnx/test_3.1_fixi"
    modify_file_path = "/home/wz/document/Art.Studio-dev-0814/onnx/test_3.1_opse"
    origin_model = onnx.load(origin_file_path)
    onnx.checker.check_model(origin_model)
    modify_model = onnx.load(modify_file_path)
    onnx.checker.check_model(modify_model)
    input_input = np.random.rand(1, 1024).astype('float32')
    input_h0 = np.random.rand(2, 1, 64).astype('float32')
    input_c0 = np.random.rand(2, 1, 64).astype('float32')
    origin_ort_sess = ort.InferenceSession(origin_file_path)
    modify_ort_sess = ort.InferenceSession(modify_file_path)
    model_input = {
    "input": input_input,
    "h0": input_h0,
    "c0": input_c0,
    }
    origin_outputs = origin_ort_sess.run(None, model_input )
    modify_outputs = modify_ort_sess.run(None, model_input )
    for i in range(len(origin_outputs)):
        print(f"result is equal? {np.array_equal(origin_outputs[i], modify_outputs[i])