print_attn_flag = False

def flag_true():
    global print_attn_flag
    print_attn_flag = True

def flag_false():
    global print_attn_flag
    print_attn_flag = False

def get_flag():
    global print_attn_flag
    return print_attn_flag
