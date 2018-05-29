def follow_spec(feat, instance):
    if feat == 'reverse':
        return instance['format'] == 'reverse_translate'
    if feat == 'listen':
        return instance['format'] == 'listen'
    return True
