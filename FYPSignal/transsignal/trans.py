#this file is to transfer datas to binary bits with multiple encoding schemes

import os
import json
import numpy as np
import jax.numpy as jnp
from docx import Document
import openpyxl

class AdvancedBinaryEncoder:
    def __init__(self):
        self.encoding_methods = {
            'direct_binary': self.direct_binary_encode,
            'manchester': self.manchester_encode,
            'nrzi': self.nrzi_encode,
            'hamming_74': self.hamming_74_encode,
            'polar_basic': self.polar_basic_encode
        }
    
    def direct_binary_encode(self, data_bytes):
        """直接二进制编码 - 每个字节转为8位二进制"""
        binary_string = ''.join(format(byte, '08b') for byte in data_bytes)
        return binary_string, "Direct binary encoding"
    
    def manchester_encode(self, data_bytes):
        """曼彻斯特编码 - 每个比特用两个比特表示: 0->01, 1->10"""
        binary_string = ''.join(format(byte, '08b') for byte in data_bytes)
        manchester_encoded = ''
        for bit in binary_string:
            if bit == '0':
                manchester_encoded += '01'
            else:
                manchester_encoded += '10'
        return manchester_encoded, "Manchester encoding (0->01, 1->10)"
    
    def nrzi_encode(self, data_bytes):
        """NRZI编码 - 电平变化表示0，不变表示1"""
        binary_string = ''.join(format(byte, '08b') for byte in data_bytes)
        nrzi_encoded = ''
        current_level = '1'  # 起始电平
        
        for bit in binary_string:
            if bit == '0':
                # 电平变化
                current_level = '0' if current_level == '1' else '1'
            # 比特为1时电平不变
            nrzi_encoded += current_level
        
        return nrzi_encoded, "NRZI encoding (transition=0, no change=1)"
    
    def hamming_74_encode(self, data_bytes):
        """汉明码(7,4) - 4个数据比特编码为7个比特，可纠正单比特错误"""
        # 汉明码生成矩阵 (4数据位 -> 7编码位)
        G = np.array([
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ])
        
        binary_string = ''.join(format(byte, '08b') for byte in data_bytes)
        hamming_encoded = ''
        
        # 每4个比特一组进行编码
        for i in range(0, len(binary_string), 4):
            data_bits = binary_string[i:i+4]
            if len(data_bits) < 4:
                data_bits = data_bits + '0' * (4 - len(data_bits))  # 填充
            
            data_vector = np.array([int(bit) for bit in data_bits])
            encoded_vector = np.dot(data_vector, G) % 2
            hamming_encoded += ''.join(str(int(bit)) for bit in encoded_vector)
        
        return hamming_encoded, "Hamming(7,4) encoding (error correction)"
    
    def polar_basic_encode(self, data_bytes, n=8):
        """基础Polar Code编码 - 使用简单的构造方法"""
        binary_string = ''.join(format(byte, '08b') for byte in data_bytes)
        
        # 简单的Polar编码实现
        def polar_transform(bits, n):
            """递归的Polar变换"""
            if len(bits) == 1:
                return bits
            # 简单的重复编码作为示例
            return bits * n
        
        # 对每个字节单独进行Polar编码
        polar_encoded = ''
        for i in range(0, len(binary_string), 8):
            byte_bits = binary_string[i:i+8]
            if len(byte_bits) == 8:
                encoded_byte = polar_transform(byte_bits, n)
                polar_encoded += encoded_byte
        
        return polar_encoded, f"Basic Polar encoding (rate 1/{n})"
    
    def encode_data(self, data_bytes, method='direct_binary', **kwargs):
        """使用指定方法编码数据"""
        if method not in self.encoding_methods:
            raise ValueError(f"Unknown encoding method: {method}")
        
        return self.encoding_methods[method](data_bytes, **kwargs)

class FileToBinaryProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.encoder = AdvancedBinaryEncoder()
        self.ensure_directories()
    
    def ensure_directories(self):
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
    
    def process_word_file(self, file_path, encoding_method='direct_binary'):
        print(f"Processing Word file: {file_path}")
        try:
            doc = Document(file_path)
            
            full_text = []
            for paragraph in doc.paragraphs:
                full_text.append(paragraph.text)
            
            for table in doc.tables:
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    full_text.append(" | ".join(row_data))
            
            text_content = "\n".join(full_text)
            data_bytes = text_content.encode('utf-8')
            
            binary_string, encoding_info = self.encoder.encode_data(data_bytes, encoding_method)
            binary_array = jnp.array([int(bit) for bit in binary_string], dtype=jnp.int32)
            
            return {
                'original_text': text_content,
                'binary_data': {
                    'binary_string': binary_string,
                    'binary_array': binary_array,
                    'total_bits': len(binary_string),
                    'encoding_method': encoding_method,
                    'encoding_info': encoding_info
                },
                'file_type': 'word',
                'file_size_bytes': len(data_bytes)
            }
            
        except Exception as e:
            print(f"Error processing Word file: {e}")
            return None
    
    def process_excel_file(self, file_path, encoding_method='direct_binary'):
        print(f"Processing Excel file: {file_path}")
        try:
            wb = openpyxl.load_workbook(file_path, data_only=True)
            
            all_data = {}
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                sheet_data = []
                for row in sheet.iter_rows(values_only=True):
                    if any(cell is not None for cell in row):
                        sheet_data.append(row)
                if sheet_data:
                    all_data[sheet_name] = sheet_data
            
            json_str = json.dumps(all_data, ensure_ascii=False)
            data_bytes = json_str.encode('utf-8')
            
            binary_string, encoding_info = self.encoder.encode_data(data_bytes, encoding_method)
            binary_array = jnp.array([int(bit) for bit in binary_string], dtype=jnp.int32)
            
            return {
                'excel_data': all_data,
                'binary_data': {
                    'binary_string': binary_string,
                    'binary_array': binary_array,
                    'total_bits': len(binary_string),
                    'encoding_method': encoding_method,
                    'encoding_info': encoding_info
                },
                'file_type': 'excel',
                'file_size_bytes': len(data_bytes)
            }
            
        except Exception as e:
            print(f"Error processing Excel file: {e}")
            return None
    
    def save_binary_data(self, file_name, processed_data):
        """保存二进制数据到文件"""
        base_name = os.path.splitext(file_name)[0]
        encoding_method = processed_data['binary_data']['encoding_method']
        
        # 1. 保存二进制字符串
        binary_txt_path = os.path.join(self.output_folder, f"{base_name}_{encoding_method}_binary.txt")
        with open(binary_txt_path, 'w', encoding='utf-8') as f:
            f.write(processed_data['binary_data']['binary_string'])
        
        # 2. 保存二进制数组
        binary_npy_path = os.path.join(self.output_folder, f"{base_name}_{encoding_method}_binary.npy")
        np.save(binary_npy_path, np.array(processed_data['binary_data']['binary_array']))
        
        # 3. 保存编码信息
        json_path = os.path.join(self.output_folder, f"{base_name}_{encoding_method}_info.json")
        json_data = {
            'file_name': file_name,
            'file_type': processed_data['file_type'],
            'file_size_bytes': processed_data['file_size_bytes'],
            'total_bits': processed_data['binary_data']['total_bits'],
            'encoding_method': encoding_method,
            'encoding_info': processed_data['binary_data']['encoding_info'],
            'binary_shape': list(processed_data['binary_data']['binary_array'].shape),
            'binary_dtype': str(processed_data['binary_data']['binary_array'].dtype)
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        
        # 4. 生成统计报告
        self.generate_statistics_report(base_name, processed_data, encoding_method)
        
        return {
            'binary_txt': binary_txt_path,
            'binary_npy': binary_npy_path,
            'info_json': json_path
        }
    
    def generate_statistics_report(self, base_name, processed_data, encoding_method):
        """生成详细的统计报告"""
        binary_array = np.array(processed_data['binary_data']['binary_array'])
        
        # 计算统计信息
        total_bits = len(binary_array)
        zeros_count = np.sum(binary_array == 0)
        ones_count = np.sum(binary_array == 1)
        zero_ratio = zeros_count / total_bits
        one_ratio = ones_count / total_bits
        
        # 计算游程统计
        def compute_run_lengths(bits):
            diffs = np.diff(bits)
            change_positions = np.where(diffs != 0)[0] + 1
            run_lengths = np.diff(np.concatenate([np.array([0]), change_positions, np.array([len(bits)])]))
            return run_lengths
        
        run_lengths = compute_run_lengths(binary_array[:min(10000, len(binary_array))])
        
        # 生成报告
        stats_path = os.path.join(self.output_folder, f"{base_name}_{encoding_method}_statistics.txt")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Binary Data Statistics Report\n")
            f.write(f"=============================\n\n")
            f.write(f"File: {base_name}\n")
            f.write(f"Type: {processed_data['file_type']}\n")
            f.write(f"Encoding: {processed_data['binary_data']['encoding_info']}\n")
            f.write(f"Original Size: {processed_data['file_size_bytes']} bytes\n")
            f.write(f"Encoded Bits: {total_bits}\n")
            f.write(f"Coding Rate: {processed_data['file_size_bytes'] * 8 / total_bits:.4f}\n\n")
            
            f.write(f"Bit Distribution:\n")
            f.write(f"  Zeros: {zeros_count} ({zero_ratio:.4f})\n")
            f.write(f"  Ones: {ones_count} ({one_ratio:.4f})\n\n")
            
            f.write(f"Run Length Statistics:\n")
            f.write(f"  Average run length: {np.mean(run_lengths):.2f}\n")
            f.write(f"  Longest run: {np.max(run_lengths)}\n")
            f.write(f"  Shortest run: {np.min(run_lengths)}\n")
            f.write(f"  Total runs: {len(run_lengths)}\n\n")
            
            f.write(f"Available Encoding Methods:\n")
            for method in self.encoder.encoding_methods.keys():
                f.write(f"  - {method}\n")
    
    def process_all_files(self, encoding_method='direct_binary'):
        """处理所有文件，使用指定的编码方法"""
        processed_files = []
        
        print(f"Using encoding method: {encoding_method}")
        available_methods = list(self.encoder.encoding_methods.keys())
        print(f"Available encoding methods: {available_methods}")
        
        for file_name in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, file_name)
            
            if os.path.isfile(file_path):
                processed_data = None
                
                if file_name.endswith('.docx'):
                    processed_data = self.process_word_file(file_path, encoding_method)
                elif file_name.endswith('.xlsx'):
                    processed_data = self.process_excel_file(file_path, encoding_method)
                
                if processed_data:
                    output_files = self.save_binary_data(file_name, processed_data)
                    
                    processed_files.append({
                        'input_file': file_name,
                        'output_files': output_files,
                        'processing_data': processed_data
                    })
                    
                    print(f"Processed: {file_name}")
                    for file_type, file_path in output_files.items():
                        print(f"  {file_type}: {os.path.basename(file_path)}")
        
        return processed_files

def main():
    # 配置路径
    input_folder = "C:/SJTU-simcom/FYPSignal/input_files"
    output_folder = "C:/SJTU-simcom/FYPSignal/output_files"
    
    processor = FileToBinaryProcessor(input_folder, output_folder)
    
    # 可选择编码方法
    encoding_methods = list(processor.encoder.encoding_methods.keys())
    
    print("Available encoding methods:")
    for i, method in enumerate(encoding_methods, 1):
        print(f"{i}. {method}")
    
    # 用户选择编码方法
    try:
        choice = input(f"\nSelect encoding method (1-{len(encoding_methods)}, default=1): ").strip()
        if choice == '':
            choice = 1
        else:
            choice = int(choice)
        
        selected_method = encoding_methods[choice-1] if 1 <= choice <= len(encoding_methods) else 'direct_binary'
    except:
        selected_method = 'direct_binary'
    
    print(f"\nStarting file processing with {selected_method} encoding...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    results = processor.process_all_files(selected_method)
    
    print(f"\nProcessing completed. Total files processed: {len(results)}")

if __name__ == "__main__":
    main()