"""Token constants for MQAR task."""

# Special token IDs
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
QUERY_TOKEN_ID = 4

# MQAR vocabulary ranges (keys/values partitioned to prevent collision)
KEY_TOKEN_START = 10
KEY_TOKEN_END = 4096
VALUE_TOKEN_START = 4096
VALUE_TOKEN_END = 8192

# Vocabulary size
MQAR_VOCAB_SIZE = 8192
MQAR_SEQ_LENGTH = 512
MAX_SEQ_LEN = 8192
