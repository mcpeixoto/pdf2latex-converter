



def escape_special_chars(s):
    """Return string s with LaTex special characters escaped."""
    special_chars = ['&', '%', '$', '#', '_', '{', '}']
    for c in special_chars:
        s = s.replace(c, '\\' + c) if c in s else s
    return s