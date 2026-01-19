def validate_file_size(uploaded_file, limit_bytes):
    if uploaded_file.size > limit_bytes:
        return False, f"File size exceeds limit of {limit_bytes/1024/1024} MB"
    return True, ""
