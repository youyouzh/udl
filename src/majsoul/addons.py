"""
打开命令行界面输入`mitmdump -p 23410 -s addons.py` 开启代理，23410是端口
将Chrome或者Edge的快捷方式 `复制->粘贴` 出现一个副本，对快捷方式副本 `右键->属性->目标` 的后面按一个空格后添加
` --proxy-server=127.0.0.1:23410 --ignore-certificate-errors https://game.maj-soul.com/1/`
代理覆盖问题： 在`mitmproxy`的启动参数中设置前置代理。例如，Clash默认端口为7890，则启动参数为：<br />`mitmdump -p 23410 -s addons.py --mode upstream:http://127.0.0.1:7890`
"""
import logging
from base64 import b64decode
import requests
import mitmproxy.http
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning
from google.protobuf.json_format import MessageToDict
import liqi
from proto import liqi_pb2 as pb

# 导入配置
SETTINGS = {
    "SEND_METHOD": [
        ".lq.Lobby.oauth2Login",
        ".lq.Lobby.fetchFriendList",
        ".lq.FastTest.authGame",
        ".lq.NotifyPlayerLoadGameReady",
        ".lq.ActionPrototype",
        ".lq.Lobby.fetchGameRecordList",
        ".lq.FastTest.syncGame",
        ".lq.Lobby.login"
    ],
    "SEND_ACTION": [
        "ActionNewRound",
        "ActionDealTile",
        "ActionAnGangAddGang",
        "ActionChiPengGang",
        "ActionNoTile",
        "ActionHule",
        "ActionBaBei",
        "ActionLiuJu",
        "ActionUnveilTile",
        "ActionHuleXueZhanMid",
        "ActionGangResult",
        "ActionRevealTile",
        "ActionChangeTile",
        "ActionSelectGap",
        "ActionLiqi",
        "ActionDiscardTile",
        "ActionHuleXueZhanEnd",
        "ActionNewCard",
        "ActionGangResultEnd"
    ],
    "API_URL": "https://localhost:12121/"
}
SEND_METHOD = SETTINGS['SEND_METHOD']  # 需要发送给小助手的method
SEND_ACTION = SETTINGS['SEND_ACTION']  # '.lq.ActionPrototype'中，需要发送给小助手的action
API_URL = SETTINGS['API_URL']  # 小助手的地址

liqi_proto = liqi.LiqiProto()
# 禁用urllib3安全警告
disable_warnings(InsecureRequestWarning)


class WebSocketAddon:

    def websocket_message(self, flow: mitmproxy.http.HTTPFlow):
        # 过滤其他请求
        # (host, port) = flow.server_conn.address
        # if host.find('game') < 0:
        #     return

        # 在捕获到WebSocket消息时触发
        assert flow.websocket is not None  # make type checker happy
        message = flow.websocket.messages[-1]
        # 解析proto消息
        result = liqi_proto.parse(message)
        if message.from_client is False:
            logging.info(f'接收到：{result}')
        if result['method'] in SEND_METHOD and message.from_client is False:
            if result['method'] == '.lq.ActionPrototype':
                if result['data']['name'] in SEND_ACTION:
                    data = result['data']['data']
                    if result['data']['name'] == 'ActionNewRound':
                        # 雀魂弃用了md5改用sha256，但没有该字段会导致小助手无法解析牌局，也不能留空
                        # 所以干脆发一个假的，反正也用不到
                        data['md5'] = data['sha256'][:32]
                else:
                    return
            elif result['method'] == '.lq.FastTest.syncGame':  # 重新进入对局时
                actions = []
                for item in result['data']['game_restore']['actions']:
                    if item['data'] == '':
                        actions.append({'name': item['name'], 'data': {}})
                    else:
                        b64 = b64decode(item['data'])
                        action_proto_obj = getattr(
                            pb, item['name']).FromString(b64)
                        action_dict_obj = MessageToDict(
                            action_proto_obj, preserving_proto_field_name=True, including_default_value_fields=True)
                        if item['name'] == 'ActionNewRound':
                            # 这里也是假md5，理由同上
                            action_dict_obj['md5'] = action_dict_obj['sha256'][:32]
                        actions.append(
                            {'name': item['name'], 'data': action_dict_obj})
                data = {'sync_game_actions': actions}
            else:
                data = result['data']
            logging.warn(f'已发送：{data}')
            requests.post(API_URL, json=data, verify=False)
            if 'liqi' in data.keys():  # 补发立直消息
                logging.warn(f'已发送：{data["liqi"]}')
                requests.post(API_URL,
                              json=data['liqi'], verify=False)


addons = [
    WebSocketAddon()
]


if __name__ == '__main__':
    # 从当前脚本启动，可以对Addon进行Debug
    from mitmproxy.tools.main import mitmdump
    mitmdump(['-p', '23410', '-s', 'addons.py'])
