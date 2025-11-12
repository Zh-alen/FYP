import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_saved_signals():
    """分析已保存的信号数据"""
    
    # 加载输入信号数据
    try:
        input_data = np.load('./results/input_signals/input_signals.npz', allow_pickle=True)
        print("输入信号数据加载成功!")
        
        # 查看可用的数据
        print("可用数据键:", list(input_data.keys()))
        
        # 分析比特序列
        if 'bit_sequences' in input_data:
            bit_sequences = input_data['bit_sequences']
            print(f"bit_sequences 形状: {bit_sequences.shape}")
            
            # 去除单例维度
            if bit_sequences.shape[0] == 1:
                bit_sequences = bit_sequences[0]  # 从 (1,2,393216) 变为 (2,393216)
            
            print(f"处理后信道数量: {bit_sequences.shape[0]}")
            
            for pol_idx in range(bit_sequences.shape[0]):
                pol_bits = bit_sequences[pol_idx]
                print(f"\n偏振 {pol_idx}:")
                print(f"  比特形状: {pol_bits.shape}")
                print(f"  前50个比特: {pol_bits[:50]}")
                
                # 统计信息
                zero_count = np.sum(pol_bits == 0)
                one_count = np.sum(pol_bits == 1)
                total_bits = len(pol_bits)
                print(f"  统计: 0={zero_count}({zero_count/total_bits*100:.1f}%), 1={one_count}({one_count/total_bits*100:.1f}%)")
        
        # 分析符号序列
        if 'symbol_sequences' in input_data:
            symbol_sequences = input_data['symbol_sequences']
            print(f"\n符号序列形状: {symbol_sequences.shape}")
            
            # 去除单例维度
            if symbol_sequences.shape[0] == 1:
                symbol_sequences = symbol_sequences[0]  # 从 (1,2,98304) 变为 (2,98304)
            
            for pol_idx in range(symbol_sequences.shape[0]):
                pol_symbols = symbol_sequences[pol_idx]
                print(f"偏振 {pol_idx} 符号形状: {pol_symbols.shape}")
                print(f"  前10个符号: {pol_symbols[:10]}")
            
            # 绘制星座图
            plot_constellation(symbol_sequences)
            
        # 分析发射波形
        if 'tx_waveforms' in input_data:
            tx_waveforms = input_data['tx_waveforms']
            print(f"\n发射波形形状: {tx_waveforms.shape}")
            
            # 绘制发射信号
            plot_tx_signals(tx_waveforms)
            
    except FileNotFoundError:
        print("输入信号文件未找到，请先运行主程序")
    except Exception as e:
        print(f"分析数据时出错: {e}")
        import traceback
        traceback.print_exc()

def plot_tx_signals(tx_waveforms):
    """绘制发射信号"""
    try:
        # 去除单例维度
        if tx_waveforms.shape[0] == 1:
            tx_waveforms = tx_waveforms[0]  # 从 (1,2,262144) 变为 (2,262144)
        
        print(f"绘图数据形状: {tx_waveforms.shape}")
        
        # 为每个偏振创建图表
        for pol_idx in range(tx_waveforms.shape[0]):
            waveform_data = tx_waveforms[pol_idx]
            print(f"偏振 {pol_idx} 波形数据形状: {waveform_data.shape}")
            
            plt.figure(figsize=(15, 10))
            
            # 绘制实部
            plt.subplot(2, 2, 1)
            real_part = np.real(waveform_data)
            # 确保x和y长度相同
            x_points = range(len(real_part[:1000]))
            y_points = real_part[:1000]
            plt.scatter(x_points, y_points, s=10, alpha=0.6, marker='.', color='blue')
            plt.title(f'Transmit Signal - Real Part (Pol {pol_idx})')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # 绘制虚部
            plt.subplot(2, 2, 2)
            imag_part = np.imag(waveform_data)
            x_points = range(len(imag_part[:1000]))
            y_points = imag_part[:1000]
            plt.scatter(x_points, y_points, s=10, alpha=0.6, marker='.', color='red')
            plt.title(f'Transmit Signal - Imaginary Part (Pol {pol_idx})')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # 绘制幅度
            plt.subplot(2, 2, 3)
            magnitude = np.abs(waveform_data)
            x_points = range(len(magnitude[:1000]))
            y_points = magnitude[:1000]
            plt.scatter(x_points, y_points, s=10, alpha=0.6, marker='.', color='green')
            plt.title(f'Transmit Signal - Magnitude (Pol {pol_idx})')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # 绘制相位
            plt.subplot(2, 2, 4)
            phase = np.angle(waveform_data)
            x_points = range(len(phase[:1000]))
            y_points = phase[:1000]
            plt.scatter(x_points, y_points, s=10, alpha=0.6, marker='.', color='purple')
            plt.title(f'Transmit Signal - Phase (Pol {pol_idx})')
            plt.xlabel('Sample Index')
            plt.ylabel('Phase (radians)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'./results/tx_signal_pol{pol_idx}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    except Exception as e:
        print(f"绘图时出错: {e}")
        import traceback
        traceback.print_exc()

def plot_constellation(symbol_sequences):
    """绘制星座图"""
    try:
        plt.figure(figsize=(12, 5))
        
        for pol_idx in range(symbol_sequences.shape[0]):
            symbols = symbol_sequences[pol_idx]
            
            plt.subplot(1, 2, pol_idx + 1)
            plt.scatter(np.real(symbols), np.imag(symbols), s=5, alpha=0.6)
            plt.title(f'Constellation - Polarization {pol_idx}')
            plt.xlabel('In-phase (I)')
            plt.ylabel('Quadrature (Q)')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig('./results/constellation_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"绘制星座图时出错: {e}")

def analyze_output_signals():
    """分析输出信号数据"""
    try:
        output_data = np.load('./results/output_signals/output_signals.npz', allow_pickle=True)
        print("\n=== 输出信号分析 ===")
        print("可用数据键:", list(output_data.keys()))
        
        # 分析原始接收信号
        if 'raw_rx_waveform' in output_data:
            raw_rx = output_data['raw_rx_waveform']
            print(f"原始接收信号形状: {raw_rx.shape}")
            
            # 去除单例维度（如果需要）
            if raw_rx.shape[0] == 1:
                raw_rx = raw_rx[0]
            
            plot_rx_signals(raw_rx, "Raw Received")
        
        # 分析处理后的信号
        if 'processed_rx_waveform' in output_data:
            processed_rx = output_data['processed_rx_waveform']
            print(f"处理后接收信号形状: {processed_rx.shape}")
            
            if processed_rx.shape[0] == 1:
                processed_rx = processed_rx[0]
            
            plot_rx_signals(processed_rx, "Processed Received")
            
    except FileNotFoundError:
        print("输出信号文件未找到")
    except Exception as e:
        print(f"分析输出信号时出错: {e}")

def plot_rx_signals(rx_waveform, title_prefix):
    """绘制接收信号"""
    try:
        for pol_idx in range(rx_waveform.shape[0]):
            waveform_data = rx_waveform[pol_idx]
            
            plt.figure(figsize=(12, 8))
            
            # 实部
            plt.subplot(2, 2, 1)
            real_part = np.real(waveform_data)
            plt.scatter(range(len(real_part[:1000])), real_part[:1000], s=10, alpha=0.6, marker='.')
            plt.title(f'{title_prefix} Signal - Real Part (Pol {pol_idx})')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # 虚部
            plt.subplot(2, 2, 2)
            imag_part = np.imag(waveform_data)
            plt.scatter(range(len(imag_part[:1000])), imag_part[:1000], s=10, alpha=0.6, marker='.')
            plt.title(f'{title_prefix} Signal - Imaginary Part (Pol {pol_idx})')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # 幅度
            plt.subplot(2, 2, 3)
            magnitude = np.abs(waveform_data)
            plt.scatter(range(len(magnitude[:1000])), magnitude[:1000], s=10, alpha=0.6, marker='.')
            plt.title(f'{title_prefix} Signal - Magnitude (Pol {pol_idx})')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # 相位
            plt.subplot(2, 2, 4)
            phase = np.angle(waveform_data)
            plt.scatter(range(len(phase[:1000])), phase[:1000], s=10, alpha=0.6, marker='.')
            plt.title(f'{title_prefix} Signal - Phase (Pol {pol_idx})')
            plt.xlabel('Sample Index')
            plt.ylabel('Phase (radians)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'./results/{title_prefix.lower().replace(" ", "_")}_pol{pol_idx}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    except Exception as e:
        print(f"绘制接收信号时出错: {e}")

if __name__ == "__main__":
    # 分析输入信号
    analyze_saved_signals()
    
    # 分析输出信号
    analyze_output_signals()