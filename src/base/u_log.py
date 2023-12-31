#!/usr/bin/python
# -*- coding: utf-8 -*
# log function

"""
:description:
    common log related module
"""
from __future__ import print_function

__all__ = [
    'debug', 'info', 'warn', 'critical',
    'init_log_instance', 'set_log_level',
    'ROTATION', 'INFINITE', 'get_inited_logger_name', 'parse'
]

import os
import re
import sys
import logging
import time
import platform
import threading
from logging import handlers


ROTATION = 0
INFINITE = 1

# log rotation count
ROTATION_COUNTS = 30

# log level
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
MIN_LEVEL = logging.DEBUG

# global logging instance name
G_INITED_LOGGER = []

# log function
debug = logging.debug
info = logging.info
warn = logging.warning
error = logging.error
critical = logging.critical


class _Singleton(object):
    """
    internal use for logging.
    """
    _LOCK = threading.Lock()

    def __init__(self, cls):
        self.__instance = None
        self.__cls = cls

    def __call__(self, *args, **kwargs):
        self._LOCK.acquire()
        if self.__instance is None:
            self.__instance = self.__cls(*args, **kwargs)
        self._LOCK.release()
        return self.__instance


class _MsgFilter(logging.Filter):
    """
    Msg filters by log levels, extends from logging.Filter
    """

    def __init__(self, msg_level=logging.WARNING):
        """
        construct function
        :param msg_level: the level of log level
        """
        super().__init__()
        self.msg_level = msg_level

    def filter(self, record):
        """
        overload, filter the msg by log level
        :param record:
        :return: true mean log, false mean don't need log
        """
        if record.levelno >= self.msg_level:
            return False
        else:
            return True


def _line(depth=0):
    """
    get current code line number
    :param depth: the depth of the frame at the top of the call stack.
    :return: line number
    """
    return sys._getframe(depth + 1).f_lineno


def _file(depth=0):
    """
    get current call function name
    :param depth: the depth of the frame at the top of the call stack.
    :return: call function name
    """
    return os.path.basename(sys._getframe(depth + 1).f_code.co_filename)


def _process_thread_id():
    """
    get current process id and thread id
    :return: a str of {process_id}:{thread_id}
    """
    return str(os.getpid()) + ':' + str(threading.current_thread().ident)


def _log_file_func_info(msg, back_trace_len=0):
    """
    log file and function info
    :param msg: message
    :param back_trace_len: back trace length
    :return: the message added file function info
    """
    temp_msg = ' * [%s] [%s:%s] ' % (
        _process_thread_id(), _file(2 + back_trace_len),
        _line(2 + back_trace_len)
    )

    msg = '%s%s' % (temp_msg, msg)
    if isinstance(msg, str):
        return msg
    else:
        return msg.decode('utf8')


def set_log_level(level):
    """
    change log level during runtime
    :param level: log level
    :return: none
    """
    logger_instance = _LoggerInstance()
    logger_instance.getlogger().setLevel(level)


def is_linux():
    """
    Check if you are running on Linux.
    :return: True or False
    """
    return platform.platform().startswith('Linux')


def is_windows():
    """
    Check if you are running on Windows.
    :return: True or False
    """
    return platform.platform().startswith('Windows')


# 日志等级过滤
class MaxLevelFilter(logging.Filter):
    """Filters (lets through) all messages with level < LEVEL"""
    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        # "<" instead of "<=": since logger.setLevel is inclusive, this should be exclusive
        return record.levelno < self.level


@_Singleton
class _LoggerInstance(object):
    """
    logger Instance object, for singleton
    """
    _logger_instance = None
    _max_size = 0
    _log_file = ''
    _log_type = ROTATION
    _print_console = False

    def __init__(self):
        pass

    def get_logger(self):
        """
        get logger instance
        :return: logger instance
        """
        if self._logger_instance is None:
            raise Exception(
                'The logger has not been initialized Yet. '
                'Call init_log_instance first'
            )
        return self._logger_instance

    def set_logger(self, logger):
        """
        set the logger instance
        :param logger: logging instance
        :return: none
        """
        if self._logger_instance is not None:
            raise Exception(
                """WARNING!!! The logger instance has been initialized already\
                .Please do NOT set twice, or call reset_log_instance replace""")
        self._logger_instance = logger

    def reset_logger(self, logger):
        """
        reset logging instance
        :param logger: logger instance
        :return: none
        """
        del self._logger_instance
        self._logger_instance = logger
        logging.root = logger

    def is_initialized(self):
        """
        judge the log instance is Initialized or not
        :return: True or False
        """
        if self._logger_instance is None:
            return False
        else:
            return True

    def config_file_logger(self, log_level, log_file, log_type,
                           max_size, print_console=True, generate_wf_file=False):
        """
        config logging instance
        :param log_level: the log level
        :param log_file: log file path
        :param log_type: log type
        :param max_size: str max size
        :param print_console: Decide whether or not print to console
        :param generate_wf_file: Decide whether or not decide warning or fetal log file
        :return: none
        """
        # if not log file path, create dir
        if not os.path.exists(log_file):
            try:
                os.mknod(log_file)
            except IOError:
                # create exception
                raise Exception(
                    'log file does not exist. '
                    'try to create it. but file creation failed'
                )

        # config object property
        self._log_file = log_file
        self._log_type = log_type
        self._logger_instance.setLevel(log_level)
        self._max_size = max_size
        self._print_console = print_console

        # '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s'
        # log format
        formatter = logging.Formatter(
            '%(levelname)s:\t %(asctime)s * '
            '[%(process)d:%(thread)x] [%(filename)s:%(lineno)s]\t %(message)s'
        )

        # print to console
        if print_console:
            info('print_console enabled, will print to stdout')
            # 避免默认basicConfig已经注册了root的StreamHandler，会重复输出日志，先移除掉
            for handler in logging.getLogger().handlers:
                if handler.name is None and isinstance(handler, logging.StreamHandler):
                    logging.getLogger().removeHandler(handler)
            # DEBUG INFO 输出到 stdout
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(formatter)
            stdout_handler.setLevel(MIN_LEVEL)
            stdout_handler.addFilter(MaxLevelFilter(WARNING))

            # WARNING 以上输出到 stderr
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setFormatter(formatter)
            stderr_handler.setLevel(max(log_level, WARNING))

            self._logger_instance.addHandler(stdout_handler)
            self._logger_instance.addHandler(stderr_handler)

        # set RotatingFileHandler
        rf_handler = None
        if log_type == ROTATION:
            rf_handler = handlers.RotatingFileHandler(self._log_file, 'a', max_size, ROTATION_COUNTS, encoding='utf-8')
        else:
            rf_handler = logging.FileHandler(self._log_file, 'a', encoding='utf-8')

        rf_handler.setFormatter(formatter)
        rf_handler.setLevel(log_level)

        # generate warning and fetal log to wf file
        if generate_wf_file:
            # add warning and fetal handler
            file_wf = str(self._log_file) + '.wf'
            warn_handler = logging.FileHandler(file_wf, 'a', encoding='utf-8')
            warn_handler.setLevel(logging.WARNING)
            warn_handler.setFormatter(formatter)
            self._logger_instance.addHandler(warn_handler)
            rf_handler.addFilter(_MsgFilter(logging.WARNING))

        self._logger_instance.addHandler(rf_handler)


def init_log_instance(name, level=logging.INFO, file='u_log', log_type=ROTATION,
                      max_size=1073741824, print_console=False, generate_wf=False):
    """
    Initialize your logging
    :param name: Unique logger name
    :param level: 4 default levels: log.DEBUG log.INFO log.ERROR log.CRITICAL
    :param file: log file. Will try to create it if no existence
    :param log_type:
        Two type: log.ROTATION and log.INFINITE

        log.ROTATION will let logfile switch to a new one (30 files at most).
        When logger reaches the 30th logfile, will overwrite from the
        oldest to the most recent.

        log.INFINITE will write on the logfile infinitely
    :param max_size: max log size with byte
    :param print_console: print to stdout or not?
    :param generate_wf: print log msg with level >= WARNING to file (${logfile}.wf)
    :return: none
    *E.g.*
    ::
        import logging
        from cup import log
        log.init_comlog(
            'test',
            log.DEBUG,
            '/home/work/test/test.log',
            log.ROTATION,
            1024,
            False
        )
        log.info('test xxx')
        log.critical('test critical')
    """
    logging_instance = _LoggerInstance()
    if not logging_instance.is_initialized():
        # initialize logging instance
        logging_instance.set_logger(logging.getLogger())

        # create log file
        if os.path.exists(file) is False:
            # create file on linux platform
            if is_linux():
                os.mknod(file)
            elif is_windows():
                with open(file, 'w+') as log_file_handle:
                    log_file_handle.write('---Windows Log File Creation ---\n')
            else:
                raise Exception("not support platform\n")
        elif os.path.isfile(file) is False:
            raise Exception('The log file exists. But it\'s not regular file\n')

        # set log config
        logging_instance.config_file_logger(level, file, log_type, max_size, print_console, generate_wf)
        info('-' * 20 + 'Log Initialized Successfully' + '-' * 20)
        # set global log name
        global G_INITED_LOGGER
        G_INITED_LOGGER.append(name)
    else:
        print('[{0}:{1}] init_log_instance has been already initialized'.format(_file(1), _line(1)))
    return


def re_init_log_instance(name, level=logging.INFO, file='u_log',
                         log_type=ROTATION, max_size=1073741824, print_console=False, generate_wf=False):
    """
    re initialize logging system, parameters same to initLogInstance.
    re_init_log_instance will reset all logging parameters，
    Make sure you used a different logger name from the old one!
    """
    global G_INITED_LOGGER
    if name in G_INITED_LOGGER:
        msg = 'logger name:%s has been already initialized!!!' % name
        raise ValueError(msg)

    logging_instance = _LoggerInstance()
    logging_instance.reset_logger(logging.getLogger(name))
    # create log file
    if os.path.exists(file) is False:
        # create file on linux platform
        if is_linux():
            os.mknod(file)
        elif is_windows():
            with open(file, 'w+') as log_file_handle:
                log_file_handle.write('---Windows Log File Creation ---\n')
        else:
            raise Exception("not support platform\n")
    elif os.path.isfile(file) is False:
        raise Exception('The log file exists. But it\'s not regular file\n')

    # set log config
    G_INITED_LOGGER.append(name)
    logging_instance.config_file_logger(level, file, log_type, max_size, print_console, generate_wf)
    info('-' * 20 + 'Log Reinitialized Successfully' + '-' * 20)
    return


def get_inited_logger_name():
    """
    get initialized logger name
    """
    global G_INITED_LOGGER
    return G_INITED_LOGGER


def _fail_handle(msg, e):
    if not isinstance(msg, str):
        msg = msg.decode('utf8')
    print('{0}\nerror:{1}'.format(msg, e))


def parse(log_line):
    """
    return a dict if the line is valid.
    Otherwise, return None
    ::
        dict_info:= {
           'loglevel': 'DEBUG',
           'date': '2015-10-14',
           'time': '16:12:22,924',
           'pid': 8808,
           'tid': 1111111,
           'srcline': 'util.py:33',
           'msg': 'this is the log content'
        }

    """
    try:
        content = log_line[log_line.find(']'):]
        content = content[(content.find(']') + 1):]
        content = content[(content.find(']') + 1):].strip()
        regex = re.compile('[ \t]+')
        items = regex.split(log_line)
        log_level, date, time_, _, pid_tid, src = items[0:6]
        pid, tid = pid_tid.strip('[]').split(':')
        return {
            'level': log_level.strip(':'),
            'date': date,
            'time': time_,
            'pid': pid,
            'tid': tid,
            'line': src.strip('[]'),
            'msg': content
        }
    except Exception:
        return None


def init_default_log(name='log'):
    # 初始化log
    log_path = os.path.join(os.getcwd(), 'log')
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    log_path = os.path.join(log_path,
                            name + '-' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.log')
    init_log_instance(name, INFO, log_path, print_console=True)


init_default_log()

if __name__ == '__main__':
    debug('中文')
    init_log_instance('test', INFO, './test.log', print_console=True)
    # test call init_log_instance twice
    init_log_instance('test', INFO, './test.log', print_console=True)

    # test log diff log level
    info('test log info')
    debug('test log debug')
    info('中文')
    error('test log error. {id}'.format(id=12))

    re_init_log_instance(
        're-test', INFO, './re.test.log',
        ROTATION, 102400000, True
    )
    info('re:test info')
    debug('re:test debug')
    debug('re:中文')
