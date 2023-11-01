import onnx
from onnx.tools import update_model_dims

def get_model_info(model):
    """打印model的meta info"""
    for field in ['doc_string', 'domain', 'functions',
    'ir_version', 'metadata_props', 'model_version',
    'opset_import', 'producer_name', 'producer_version',
    'training_info']:
        print(f"{field}: {getattr(model, field)}")

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

if __name__ == "__main__":
    onnx_path = r"/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/model_0364999-dy.onnx"
    model = onnx.load(onnx_path)
    get_model_info(model)
    pretty_print_model_info(model)