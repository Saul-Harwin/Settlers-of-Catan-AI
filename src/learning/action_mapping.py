def decode_action(action_id):
    if action_id < 72:
        return ("build_road", action_id)

    elif action_id < 126:
        return ("build_settlement", action_id - 72)

    elif action_id < 180:
        return ("build_city", action_id - 126)

    elif action_id == 180:
        return ("end_turn", None)

    else:
        trade_index = action_id - 181
        give = trade_index // 4
        receive_adj = trade_index % 4

        receive = receive_adj if receive_adj < give else receive_adj + 1

        return ("trade_4to1", (give, receive))