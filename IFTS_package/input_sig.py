import torch
import numpy as np
import os
from datetime import datetime

class InputSignalExtractor:
    def __init__(self, save_path='./results/input_signals/'):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
    def extract_tx_signals(self, bit_seq, sym_seq, tx_seq, integer_seq, sym_map, sig_para, tx_para):
        """
        提取所有输入信号数据
        """
        print("=== 提取输入信号数据 ===")
        
        input_data = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'bit_sequences': self._process_bit_sequences(bit_seq),
            'symbol_sequences': self._process_symbol_sequences(sym_seq),
            'tx_sequences': self._process_tx_sequences(tx_seq),
            'integer_sequences': self._process_integer_sequences(integer_seq),
            'constellation_map': self._process_constellation_map(sym_map),
            'signal_parameters': self._extract_signal_parameters(sig_para, tx_para)
        }
        
        # 保存数据
        self._save_input_data(input_data)
        return input_data
    
    def _process_bit_sequences(self, bit_seq):
        """处理比特序列"""
        print("处理比特序列...")
        bit_data = {}
        
        for i_ch, channel_bits in enumerate(bit_seq):
            channel_data = {}
            for i_pol, pol_bits in enumerate(channel_bits):
                if torch.is_tensor(pol_bits):
                    bits = pol_bits.cpu().numpy() if pol_bits.is_cuda else pol_bits.numpy()
                else:
                    bits = np.array(pol_bits)
                
                channel_data[f'pol_{i_pol}'] = {
                    'shape': bits.shape,
                    'dtype': str(bits.dtype),
                    'total_bits': bits.size,
                    'bit_statistics': {
                        'zeros_count': np.sum(bits == 0),
                        'ones_count': np.sum(bits == 1),
                        'zero_ratio': np.mean(bits == 0)
                    },
                    'data': bits
                }
            bit_data[f'channel_{i_ch}'] = channel_data
        
        return bit_data
    
    def _process_symbol_sequences(self, sym_seq):
        """处理符号序列"""
        print("处理符号序列...")
        sym_data = {}
        
        for i_ch, channel_symbols in enumerate(sym_seq):
            channel_data = {}
            for i_pol, pol_symbols in enumerate(channel_symbols):
                if torch.is_tensor(pol_symbols):
                    symbols = pol_symbols.cpu().numpy() if pol_symbols.is_cuda else pol_symbols.numpy()
                else:
                    symbols = np.array(pol_symbols)
                
                # 转换为复数形式（如果是I/Q分量的情况）
                if symbols.ndim > 1 and symbols.shape[-1] == 2:
                    complex_symbols = symbols[..., 0] + 1j * symbols[..., 1]
                else:
                    complex_symbols = symbols
                
                channel_data[f'pol_{i_pol}'] = {
                    'shape': complex_symbols.shape,
                    'dtype': str(complex_symbols.dtype),
                    'total_symbols': complex_symbols.size,
                    'symbol_statistics': {
                        'average_power': np.mean(np.abs(complex_symbols)**2),
                        'mean_real': np.mean(np.real(complex_symbols)),
                        'mean_imag': np.mean(np.imag(complex_symbols)),
                        'constellation_points': np.unique(complex_symbols)
                    },
                    'data': complex_symbols
                }
            sym_data[f'channel_{i_ch}'] = channel_data
        
        return sym_data
    
    def _process_tx_sequences(self, tx_seq):
        """处理发射序列"""
        print("处理发射序列...")
        tx_data = {}
        
        for i_ch, channel_tx in enumerate(tx_seq):
            if torch.is_tensor(channel_tx):
                tx_signal = channel_tx.cpu().numpy() if channel_tx.is_cuda else channel_tx.numpy()
            else:
                tx_signal = np.array(channel_tx)
            
            tx_data[f'channel_{i_ch}'] = {
                'shape': tx_signal.shape,
                'dtype': str(tx_signal.dtype),
                'signal_statistics': {
                    'average_power': np.mean(np.abs(tx_signal)**2),
                    'peak_power': np.max(np.abs(tx_signal)**2),
                    'papr': np.max(np.abs(tx_signal)**2) / np.mean(np.abs(tx_signal)**2) if np.mean(np.abs(tx_signal)**2) > 0 else 0
                },
                'data': tx_signal
            }
        
        return tx_data
    
    def _process_integer_sequences(self, integer_seq):
        """处理整数序列"""
        print("处理整数序列...")
        int_data = {}
        
        for i_ch, channel_ints in enumerate(integer_seq):
            channel_data = {}
            for i_pol, pol_ints in enumerate(channel_ints):
                if torch.is_tensor(pol_ints):
                    integers = pol_ints.cpu().numpy() if pol_ints.is_cuda else pol_ints.numpy()
                else:
                    integers = np.array(pol_ints)
                
                channel_data[f'pol_{i_pol}'] = {
                    'shape': integers.shape,
                    'dtype': str(integers.dtype),
                    'unique_values': np.unique(integers),
                    'data': integers
                }
            int_data[f'channel_{i_ch}'] = channel_data
        
        return int_data
    
    def _process_constellation_map(self, sym_map):
        """处理星座图映射"""
        print("处理星座图映射...")
        constellation_data = {}
        
        for i, mapping in enumerate(sym_map):
            if torch.is_tensor(mapping):
                constellation = mapping.cpu().numpy() if mapping.is_cuda else mapping.numpy()
            else:
                constellation = np.array(mapping)
            
            constellation_data[f'map_{i}'] = {
                'shape': constellation.shape,
                'constellation_points': constellation,
                'modulation_order': len(constellation)
            }
        
        return constellation_data
    
    def _extract_signal_parameters(self, sig_para, tx_para):
        """提取信号参数"""
        params = {
            'channel_num': getattr(sig_para, 'channel_num', 'N/A'),
            'nPol': getattr(sig_para, 'nPol', 'N/A'),
            'sample_rate': getattr(sig_para, 'sample_rate', 'N/A'),
            'symbol_rate': getattr(sig_para, 'symbol_rate', 'N/A'),
            'modulation': getattr(sig_para, 'modulation', 'N/A'),
            'tx_power': getattr(tx_para, 'power_dbm', 'N/A')
        }
        return params
    
    def _save_input_data(self, input_data):
        """保存输入数据到文件"""
        timestamp = input_data['timestamp']
        
        # 保存为NumPy格式
        np.savez(
            f"{self.save_path}input_signals_{timestamp}.npz",
            bit_sequences=input_data['bit_sequences'],
            symbol_sequences=input_data['symbol_sequences'],
            tx_sequences=input_data['tx_sequences'],
            integer_sequences=input_data['integer_sequences']
        )
        
        # 保存文本摘要
        with open(f"{self.save_path}input_summary_{timestamp}.txt", 'w') as f:
            f.write("输入信号数据摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"信道数量: {len(input_data['bit_sequences'])}\n")
            f.write(f"偏振数量: {input_data['signal_parameters']['nPol']}\n")
            f.write(f"调制方式: {input_data['signal_parameters']['modulation']}\n")
        
        print(f"输入信号数据已保存到: {self.save_path}")

def main():
    """使用示例"""
    # 这里需要你传入实际的信号数据
    extractor = InputSignalExtractor()
    # input_data = extractor.extract_tx_signals(bit_seq, sym_seq, tx_seq, integer_seq, sym_map, sig_para, tx_para)
    print("Input signal extractor ready!")

if __name__ == "__main__":
    main()