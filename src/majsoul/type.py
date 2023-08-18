from enum import Enum


# websocket中的操作类型枚举
class Operation(Enum):
    NO_EFFECT = 0  # 没有影响，比如发表情等
    DISCARD = 1  # 出牌
    CHI = 2    # 吃
    PENG = 3   # 碰
    MING_GANG = 5  # 明杠
    JIA_GANG = 6   # 加杠
    LIQI = 7  # 立直
    ZIMO = 8  # 自摸
    HU = 9    # 胡
