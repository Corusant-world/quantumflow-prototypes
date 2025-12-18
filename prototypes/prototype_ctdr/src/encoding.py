"""
Encoding utilities for short2 DPX operations
Converts strings to short2 format for DPX_LCP_Kernel
"""

import struct


def encode_to_short2(text: str) -> bytes:
    """
    Кодирование текста в short2 для DPX.
    
    Схема:
    - Каждый символ → 16-битное значение (ord(char))
    - Пара символов → short2 (x, y) = 4 bytes (little-endian)
    - Padding для нечетной длины (последний символ + 0)
    
    Args:
        text: Входная строка
        
    Returns:
        bytes: Закодированные данные (каждая пара символов = 4 bytes)
    """
    if not text:
        return b''
    
    encoded = bytearray()
    for i in range(0, len(text), 2):
        if i + 1 < len(text):
            x = ord(text[i])
            y = ord(text[i + 1])
        else:
            x = ord(text[i])
            y = 0  # Padding для нечетной длины
        
        # Pack as two unsigned shorts (16-bit each) in little-endian
        # short2 is represented as 4 bytes: [x_low, x_high, y_low, y_high]
        encoded.extend(struct.pack('<HH', x, y))
    
    return bytes(encoded)


def decode_from_short2(data: bytes) -> str:
    """
    Декодирование из short2 обратно в строку.
    
    Args:
        data: Закодированные данные (должно быть кратно 4 bytes)
        
    Returns:
        str: Декодированная строка
    """
    if not data:
        return ''
    
    if len(data) % 4 != 0:
        raise ValueError(f"Data length must be multiple of 4, got {len(data)}")
    
    decoded = []
    for i in range(0, len(data), 4):
        # Unpack two unsigned shorts (16-bit each) from little-endian
        x, y = struct.unpack('<HH', data[i:i+4])
        
        # Add characters (skip padding: y == 0 at the end)
        decoded.append(chr(x))
        if y != 0:
            decoded.append(chr(y))
    
    return ''.join(decoded)


