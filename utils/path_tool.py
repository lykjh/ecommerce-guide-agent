import os

def get_pro_root() ->str:
    """
    获取项目根目录
    :return:
    """

    #必须用 __file__ 定位：os.path.abspath("path_tool.py") 是按「当前工作目录」解析的，
    #换个目录运行脚本就会算错根目录（实际会多退一级），导致 config/*.yml 找不到
    current_file=os.path.abspath(__file__)
    #utils/ 所在目录
    current_dir=os.path.dirname(current_file)
    #utils/ 的上一级 = 项目根目录
    pro_root=os.path.dirname(current_dir)
    return pro_root


def get_abs_path(relative_path) ->str:
    """
    获取绝对路径
    :param relative_path:
    :return:
    """

    project_root=get_pro_root()
    return os.path.join(project_root,relative_path)




if __name__ == '__main__':
    print(get_pro_root())