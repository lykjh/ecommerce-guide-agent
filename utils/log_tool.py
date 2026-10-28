import logging
import os
import sys
from datetime import datetime
from utils.path_tool import get_abs_path

LOG_ROOT = get_abs_path("logs")

os.makedirs(LOG_ROOT,exist_ok=True)

#日志正文格式（控制台和文件共用）
LOG_FMT="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

#ANSI 颜色码
COLOR_RESET="\033[0m"
COLOR_GRAY="\033[90m"     #DEBUG：灰色，调试信息最不重要，压暗处理
COLOR_GREEN="\033[32m"    #INFO：绿色，正常流程
COLOR_YELLOW="\033[33m"   #WARNING：黄色，需要注意但没坏
COLOR_RED="\033[31m"      #ERROR / CRITICAL：红色，出问题了

#日志级别 -> 颜色
LEVEL_COLORS={
    logging.DEBUG:COLOR_GRAY,
    logging.INFO:COLOR_GREEN,
    logging.WARNING:COLOR_YELLOW,
    logging.ERROR:COLOR_RED,
    logging.CRITICAL:COLOR_RED,
}

def _enable_windows_color():
    """
    Windows 老版控制台默认不解析 ANSI 转义序列，这里手动开启「虚拟终端处理」。
    Windows 10 及以上支持；失败就静默跳过 —— 最坏结果只是没颜色，不影响日志本身。
    """
    if sys.platform!="win32":
        return
    try:
        import ctypes
        kernel32=ctypes.windll.kernel32
        #-12 = STD_ERROR_HANDLE（StreamHandler 默认走 stderr）
        #7 = ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-12),7)
    except Exception:
        pass

_enable_windows_color()


class ColorFormatter(logging.Formatter):
    """控制台专用格式器：按日志级别给整行上色"""
    def __init__(self,fmt,use_color:bool=True):
        super().__init__(fmt)
        self.use_color=use_color

    def format(self,record):
        #先按普通方式生成文本
        msg=super().format(record)
        #不上色就直接返回（输出被重定向到文件或管道时走这里）
        if not self.use_color:
            return msg
        color=LEVEL_COLORS.get(record.levelno,COLOR_RESET)
        return f"{color}{msg}{COLOR_RESET}"


#文件用：普通格式器，绝不能上色，否则 ANSI 转义码会写进日志文件变成乱码
DEFAULT_LOGGING_FORMAT=logging.Formatter(LOG_FMT)

#控制台用：只有真的输出到终端才上色。
#stderr 被重定向到文件或管道时 isatty() 为 False，此时自动不上色
CONSOLE_LOGGING_FORMAT=ColorFormatter(
    LOG_FMT,
    use_color=getattr(sys.stderr,"isatty",lambda:False)(),
)

def get_loger(
        name:str="agent",
        console_level:int=logging.DEBUG,
        file_level:int=logging.DEBUG,
        log_file=None,
) ->logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler,"_from_log_tool",False):
            logger.removeHandler(handler)
            handler.close()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    #控制台用带颜色的格式器
    console_handler.setFormatter(CONSOLE_LOGGING_FORMAT)
    console_handler._from_log_tool = True
    logger.addHandler(console_handler)



    if not log_file:
        today = datetime.now().strftime("%Y-%m-%d")
        log_file=os.path.join(LOG_ROOT,f"{name}_{today}.log")
    else:
        if not os.path.isabs(log_file):
            log_file = os.path.join(LOG_ROOT, log_file)

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_handler= logging.FileHandler(log_file,encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOGGING_FORMAT)
    file_handler._from_log_tool = True
    logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    # 测试一下
    logger = get_loger("test_logger", console_level=logging.DEBUG, file_level=logging.DEBUG)
    logger.info("这是一条 INFO 日志，会打印在控制台并写入文件")
