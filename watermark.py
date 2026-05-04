"""
WATERMARK SYSTEM - QIM METHOD
Execute from terminal: python watermark.py
"""

import numpy as np
import cv2
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import sys
import time
import argparse

class WatermarkSystem:
    def __init__(self, delta=50):
        self.delta = delta
        self.key = None
        self.watermark_bits = None
        
    def _dct2(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _idct2(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def _quantize(self, coeff, bit):
        if bit == 0:
            return np.round(coeff / self.delta) * self.delta
        else:
            return (np.round((coeff - self.delta/2) / self.delta) * self.delta) + self.delta/2
    
    def _extract_bit(self, coeff):
        q = np.round(coeff / self.delta)
        return int(q % 2)
    
    def message_to_bits(self, message):
        if isinstance(message, str):
            binary = ''.join(format(ord(c), '08b') for c in message)
            return [int(bit) for bit in binary]
        return message
    
    def bits_to_message(self, bits):
        binary_string = ''.join(str(bit) for bit in bits)
        chars = []
        for i in range(0, len(binary_string), 8):
            if i + 8 <= len(binary_string):
                byte = binary_string[i:i+8]
                chars.append(chr(int(byte, 2)))
        return ''.join(chars)
    
    def embed(self, image_path, message, key=42):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        self.key = key
        np.random.seed(key)
        
        h, w = img.shape
        img_watermarked = img.copy().astype(np.float64)
        
        blocks_h, blocks_w = h // 8, w // 8
        total_blocks = blocks_h * blocks_w
        
        bits = self.message_to_bits(message)
        
        if len(bits) > total_blocks:
            raise ValueError(f"Message too long: {len(bits)} bits > {total_blocks} blocks")
        
        positions = np.random.permutation(total_blocks)
        self.watermark_bits = bits
        
        bit_index = 0
        block_index = 0
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                if bit_index >= len(bits):
                    break
                
                if block_index in positions[:len(bits)]:
                    block = img[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float64)
                    dct_block = self._dct2(block)
                    quantized_coeff = self._quantize(dct_block[4, 4], bits[bit_index])
                    dct_block[4, 4] = quantized_coeff
                    block_watermarked = self._idct2(dct_block)
                    img_watermarked[i*8:(i+1)*8, j*8:(j+1)*8] = block_watermarked
                    bit_index += 1
                
                block_index += 1
        
        img_watermarked = np.clip(img_watermarked, 0, 255).astype(np.uint8)
        
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_watermarked{ext}"
        cv2.imwrite(output_path, img_watermarked)
        
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        psnr_value = psnr(original_img, img_watermarked)
        
        return output_path, psnr_value, len(bits)
    
    def extract(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        if self.key is None:
            raise ValueError("Key not set. Use embed first or set key manually.")
        
        np.random.seed(self.key)
        
        h, w = img.shape
        blocks_h, blocks_w = h // 8, w // 8
        total_blocks = blocks_h * blocks_w
        
        positions = np.random.permutation(total_blocks)
        
        extracted_bits = []
        bit_index = 0
        block_index = 0
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                if self.watermark_bits and bit_index >= len(self.watermark_bits):
                    break
                
                if block_index in positions[:len(self.watermark_bits) if self.watermark_bits else total_blocks]:
                    block = img[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float64)
                    dct_block = self._dct2(block)
                    bit = self._extract_bit(dct_block[4, 4])
                    extracted_bits.append(bit)
                    bit_index += 1
                
                block_index += 1
        
        message = None
        if self.watermark_bits and len(extracted_bits) == len(self.watermark_bits):
            try:
                message = self.bits_to_message(extracted_bits)
            except:
                message = "Decoding error"
        
        return extracted_bits, message
    
    def calculate_ber(self, original_bits, extracted_bits):
        min_len = min(len(original_bits), len(extracted_bits))
        errors = np.sum(original_bits[:min_len] != extracted_bits[:min_len])
        return errors / min_len if min_len > 0 else 1.0
    
    def apply_attack(self, image_path, attack_type, output_path=None, **kwargs):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if attack_type == 'gaussian_noise':
            std = kwargs.get('std', 25)
            noise = np.random.normal(0, std, img.shape)
            img_attacked = np.clip(img + noise, 0, 255).astype(np.uint8)
            
        elif attack_type == 'jpeg':
            quality = kwargs.get('quality', 50)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            img_attacked = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
            
        elif attack_type == 'blur':
            kernel_size = kwargs.get('kernel_size', 3)
            img_attacked = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
        else:
            raise ValueError(f"Unknown attack: {attack_type}")
        
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_{attack_type}{ext}"
        
        cv2.imwrite(output_path, img_attacked)
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Watermark System - QIM Method')
    parser.add_argument('command', choices=['embed', 'extract', 'test', 'compare'],
                       help='Command to execute')
    parser.add_argument('--input', '-i', help='Input image path')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--message', '-m', help='Message to hide')
    parser.add_argument('--key', '-k', type=int, default=42, help='Secret key (default: 42)')
    parser.add_argument('--delta', '-d', type=int, default=50, help='Delta value (default: 50)')
    parser.add_argument('--attack', '-a', choices=['gaussian', 'jpeg', 'blur'],
                       help='Attack type for test')
    parser.add_argument('--quality', '-q', type=int, default=50, help='JPEG quality (1-100)')
    parser.add_argument('--std', type=int, default=25, help='Noise standard deviation')
    
    args = parser.parse_args()
    
    if args.command == 'embed':
        print("\n" + "="*60)
        print("  EMBED WATERMARK")
        print("="*60)
        
        if not args.input:
            args.input = input("Image path: ").strip()
        if not args.message:
            args.message = input("Message to hide: ").strip()
        
        print(f"\nProcessing...")
        wm = WatermarkSystem(delta=args.delta)
        
        start = time.time()
        output, psnr_val, bits = wm.embed(args.input, args.message, args.key)
        elapsed = time.time() - start
        
        print(f"\n✅ SUCCESS!")
        print(f"   Output: {output}")
        print(f"   PSNR: {psnr_val:.2f} dB")
        print(f"   Bits: {bits}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Key: {args.key}")
        
    elif args.command == 'extract':
        print("\n" + "="*60)
        print("  EXTRACT WATERMARK")
        print("="*60)
        
        if not args.input:
            args.input = input("Watermarked image path: ").strip()
        
        print(f"\nExtracting...")
        wm = WatermarkSystem(delta=args.delta)
        wm.key = args.key
        
        # Need to know original message length
        msg_len = input("Original message length (characters): ").strip()
        try:
            wm.watermark_bits = [0] * (int(msg_len) * 8)
        except:
            wm.watermark_bits = [0] * 32
        
        bits, message = wm.extract(args.input)
        
        print(f"\n✅ EXTRACTION:")
        if message:
            print(f"   Message: '{message}'")
        else:
            print(f"   Bits extracted: {len(bits)}")
            print(f"   First 32 bits: {''.join(str(b) for b in bits[:32])}")
        
    elif args.command == 'test':
        print("\n" + "="*60)
        print("  ROBUSTNESS TEST")
        print("="*60)
        
        if not args.input:
            args.input = input("Watermarked image path: ").strip()
        
        print("\nAvailable attacks:")
        print("  1. Gaussian noise")
        print("  2. JPEG compression")
        print("  3. Blur")
        
        choice = input("\nChoose attack (1-3): ").strip()
        
        wm = WatermarkSystem(delta=args.delta)
        wm.key = args.key
        
        # Get original message
        msg = input("Original message: ").strip()
        original_bits = wm.message_to_bits(msg)
        wm.watermark_bits = original_bits
        
        if choice == '1':
            std = input("Noise level (std, default 25): ").strip()
            std = int(std) if std else 25
            attacked = wm.apply_attack(args.input, 'gaussian_noise', std=std)
            print(f"\n   Applied Gaussian noise (std={std})")
        elif choice == '2':
            quality = input("JPEG quality (1-100, default 50): ").strip()
            quality = int(quality) if quality else 50
            attacked = wm.apply_attack(args.input, 'jpeg', quality=quality)
            print(f"\n   Applied JPEG compression (quality={quality})")
        elif choice == '3':
            kernel = input("Blur kernel size (3 or 5, default 3): ").strip()
            kernel = int(kernel) if kernel else 3
            attacked = wm.apply_attack(args.input, 'blur', kernel_size=kernel)
            print(f"\n   Applied Gaussian blur (kernel={kernel}x{kernel})")
        else:
            print("Invalid choice")
            return
        
        bits, message = wm.extract(attacked)
        ber = wm.calculate_ber(original_bits, bits)
        
        print(f"\n📊 RESULTS:")
        print(f"   Extracted message: '{message if message else '???'}'")
        print(f"   BER: {ber:.4f} ({ber*100:.2f}%)")
        
        if ber == 0:
            print("   ✅ Perfect! Watermark intact.")
        elif ber < 0.1:
            print("   Very robust.")
        elif ber < 0.3:
            print("    Moderately robust.")
        else:
            print("   Fragile.")
        
        print(f"   Attacked image: {attacked}")
        
    elif args.command == 'compare':
        print("\n" + "="*60)
        print("  COMPARE IMAGES (PSNR)")
        print("="*60)
        
        img1 = input("Original image path: ").strip()
        img2 = input("Watermarked image path: ").strip()
        
        orig = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        wm_img = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
        
        psnr_val = psnr(orig, wm_img)
        
        print(f"\n📊 PSNR: {psnr_val:.2f} dB")
        
        if psnr_val > 40:
            print("   Quality: EXCELLENT - Invisible watermark")
        elif psnr_val > 35:
            print("   Quality: VERY GOOD - Almost invisible")
        elif psnr_val > 30:
            print("   Quality: GOOD - Slightly visible")
        else:
            print("   Quality: POOR - Visible degradation")


if __name__ == "__main__":
    main()