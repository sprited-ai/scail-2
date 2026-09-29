from unittest.mock import Mock, patch
import pytest
from comfy_client import ComfyServer


def test_lost_websocket_stops_gpu_server():
    server = ComfyServer('/unused', '/unused', '/unused', '/unused')
    server.alive = Mock(return_value=True)
    server.post = Mock(return_value={'prompt_id': 'job'})
    server.stop = Mock()
    ws = Mock()
    ws.recv.side_effect = ConnectionError('lost websocket')
    with patch('comfy_client.websocket.create_connection', return_value=ws):
        with pytest.raises(ConnectionError):
            server.run({})
    server.stop.assert_called_once()
    ws.close.assert_called_once()
