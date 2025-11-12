import torch
import numpy as np
import os
from datetime import datetime

class OutputSignalExtractor:
    def __init__(self, save_path='./results/output_signals/'):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
    def extract_rx_signals(self, rx_sig, rx_sig_after_dsp, sig_noise, sig_int, sig_para, rx_para, performance_metrics):
        """
        提取所有输出信号数据
        """
        print("=== 提取输出信号数据 ===")
        
        output_data = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'raw_received_signal': self._process_raw_rx_signal(rx_sig),
            'processed_received_signal': self._process_processed_rx_signal(rx_sig_after_dsp),
            'noise_signal': self._process_noise_signal(sig_noise),
            'demodulated_signal': self._process_demodulated_signal(sig_int),
            'performance_metrics': performance_metrics,
            'receiver_parameters': self._extract_receiver_parameters(sig_para, rx_para)
        }
        
        # 保存数据
        self._save_output_data(output_data)
        return output_data
    
    def _process_raw_rx_signal(self, rx_sig):
        """处理原始接收信号"""
        print("处理原始接收信号...")
        
        if torch.is_tensor(rx_sig):
            raw_signal = rx_sig.cpu().numpy() if rx_sig.is_cuda else rx_sig.numpy()
        else:
            raw_signal = np.array(rx_sig)
        
        return {
            'shape': raw_signal.shape,
            'dtype': str(raw_signal.dtype),
            'signal_statistics': {
                'average_power': np.mean(np.abs(raw_signal)**2),
                'peak_power': np.max(np.abs(raw_signal)**2),
                'dynamic_range': np.max(np.abs(raw_signal)) - np.min(np.abs(raw_signal)),
                'mean_real': np.mean(np.real(raw_signal)) if np.iscomplexobj(raw_signal) else np.mean(raw_signal),
                'mean_imag': np.mean(np.imag(raw_signal)) if np.iscomplexobj(raw_signal) else 0
            },
            'data': raw_signal
        }
    
    def _process_processed_rx_signal(self, rx_sig_after_dsp):
        """处理DSP后的接收信号"""
        print("处理DSP后接收信号...")
        
        if torch.is_tensor(rx_sig_after_dsp):
            processed_signal = rx_sig_after_dsp.cpu().numpy() if rx_sig_after_dsp.is_cuda else rx_sig_after_dsp.numpy()
        else:
            processed_signal = np.array(rx_sig_after_dsp)
        
        return {
            'shape': processed_signal.shape,
            'dtype': str(processed_signal.dtype),
            'signal_statistics': {
                'average_power': np.mean(np.abs(processed_signal)**2),
                'evm': self._calculate_evm(processed_signal),  # 误差矢量幅度
                'snr_estimate': self._estimate_snr(processed_signal)
            },
            'data': processed_signal
        }
    
    def _process_noise_signal(self, sig_noise):
        """处理噪声信号"""
        print("处理噪声信号...")
        noise_data = {}
        
        if isinstance(sig_noise, (list, tuple)):
            for i, noise in enumerate(sig_noise):
                if torch.is_tensor(noise):
                    noise_array = noise.cpu().numpy() if noise.is_cuda else noise.numpy()
                else:
                    noise_array = np.array(noise)
                
                noise_data[f'pol_{i}'] = {
                    'shape': noise_array.shape,
                    'noise_statistics': {
                        'noise_power': np.mean(np.abs(noise_array)**2),
                        'noise_variance': np.var(noise_array),
                        'snr': 10 * np.log10(1 / np.var(noise_array)) if np.var(noise_array) > 0 else 0
                    },
                    'data': noise_array
                }
        else:
            if torch.is_tensor(sig_noise):
                noise_array = sig_noise.cpu().numpy() if sig_noise.is_cuda else sig_noise.numpy()
            else:
                noise_array = np.array(sig_noise)
            
            noise_data['combined'] = {
                'shape': noise_array.shape,
                'noise_statistics': {
                    'noise_power': np.mean(np.abs(noise_array)**2),
                    'noise_variance': np.var(noise_array)
                },
                'data': noise_array
            }
        
        return noise_data
    
    def _process_demodulated_signal(self, sig_int):
        """处理解调后的信号"""
        print("处理解调信号...")
        
        if torch.is_tensor(sig_int):
            demodulated = sig_int.cpu().numpy() if sig_int.is_cuda else sig_int.numpy()
        else:
            demodulated = np.array(sig_int)
        
        return {
            'shape': demodulated.shape,
            'dtype': str(demodulated.dtype),
            'demodulation_statistics': {
                'unique_symbols': np.unique(demodulated),
                'symbol_count': len(demodulated)
            },
            'data': demodulated
        }
    
    def _calculate_evm(self, signal):
        """计算误差矢量幅度"""
        if np.iscomplexobj(signal):
            ideal_power = np.mean(np.abs(signal)**2)
            if ideal_power > 0:
                # 简化的EVM计算，实际可能需要参考星座点
                return np.sqrt(np.mean(np.abs(signal - np.mean(signal))**2) / ideal_power)
        return 0
    
    def _estimate_snr(self, signal):
        """估计信噪比"""
        if np.iscomplexobj(signal):
            signal_power = np.mean(np.abs(signal)**2)
            noise_power = np.var(signal)
            if noise_power > 0:
                return 10 * np.log10(signal_power / noise_power)
        return 0
    
    def _extract_receiver_parameters(self, sig_para, rx_para):
        """提取接收机参数"""
        params = {
            'cut_idx': getattr(rx_para, 'cut_idx', 'N/A'),
            'bandwidth': getattr(rx_para, 'bandwidth', 'N/A'),
            'equalizer_type': getattr(rx_para, 'eq_type', 'N/A'),
            'ber': getattr(sig_para, 'ber_array', [0, 0]),
            'gmi': getattr(sig_para, 'gmi_value', 'N/A'),
            'mi': getattr(sig_para, 'mi_value', 'N/A')
        }
        return params
    
    def _save_output_data(self, output_data):
        """保存输出数据到文件"""
        timestamp = output_data['timestamp']
        
        # 保存为NumPy格式
        np.savez(
            f"{self.save_path}output_signals_{timestamp}.npz",
            raw_rx_signal=output_data['raw_received_signal']['data'],
            processed_rx_signal=output_data['processed_received_signal']['data'],
            noise_signal=output_data['noise_signal'],
            demodulated_signal=output_data['demodulated_signal']['data']
        )
        
        # 保存性能指标
        with open(f"{self.save_path}performance_{timestamp}.txt", 'w') as f:
            f.write("输出信号性能指标\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"偏振X BER: {output_data['performance_metrics'].get('ber_x', 'N/A'):.6f}\n")
            f.write(f"偏振Y BER: {output_data['performance_metrics'].get('ber_y', 'N/A'):.6f}\n")
            f.write(f"GMI: {output_data['performance_metrics'].get('gmi', 'N/A'):.6f}\n")
            f.write(f"MI: {output_data['performance_metrics'].get('mi', 'N/A'):.6f}\n")
        
        print(f"输出信号数据已保存到: {self.save_path}")

def main():
    """使用示例"""
    extractor = OutputSignalExtractor()
    print("Output signal extractor ready!")

if __name__ == "__main__":
    main()