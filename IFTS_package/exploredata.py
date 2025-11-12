import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def explore_input_signals():
    print("explore the input signal...")
    
    # 检查文件路径
    file_path = './results/input_signals/input_signals.npz'
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        if os.path.exists('./results/'):
            print("Contents of results directory:")
            for root, dirs, files in os.walk('./results/'):
                for file in files:
                    print(f"  {os.path.join(root, file)}")
        return
    
    data = np.load(file_path)
    
    print("usable data keys:", list(data.keys()))
    
    # 分析比特序列
    if 'bit_sequences' in data:
        bits = data['bit_sequences']
        print(f"\nbit sequence shape: {bits.shape}")
        
        # 去除单例维度
        if bits.shape[0] == 1:
            bits = bits[0]
        
        # 3D可视化比特序列
        plot_bits_3d(bits)
    
    # 分析符号序列
    if 'symbol_sequences' in data:
        symbols = data['symbol_sequences']
        print(f"\nsymbol sequence shape: {symbols.shape}")
        
        if symbols.shape[0] == 1:
            symbols = symbols[0]
        
        # 3D星座图
        plot_constellation_3d(symbols)
        
        # 传统2D星座图对比
        plot_constellation_2d(symbols)
    
    # 分析发射波形
    if 'tx_waveforms' in data:
        tx_waveforms = data['tx_waveforms']
        print(f"\ntx waveforms shape: {tx_waveforms.shape}")
        
        if tx_waveforms.shape[0] == 1:
            tx_waveforms = tx_waveforms[0]
        
        # 3D发射信号可视化
        plot_tx_signals_3d(tx_waveforms)

def plot_bits_3d(bits):
    """在3D空间中可视化比特序列"""
    fig = plt.figure(figsize=(15, 10))
    
    for pol_idx in range(bits.shape[0]):
        pol_bits = bits[pol_idx]
        
        # 创建3D子图
        ax = fig.add_subplot(2, 2, pol_idx + 1, projection='3d')
        
        # 生成3D坐标
        n_samples = min(500, len(pol_bits))
        x = range(n_samples)  # 时间轴
        y = pol_bits[:n_samples]  # 比特值 (0或1)
        z = np.zeros(n_samples)  # 偏振维度
        
        # 根据比特值着色
        colors = ['red' if bit == 1 else 'blue' for bit in y]
        
        # 绘制3D散点图
        scatter = ax.scatter(x, y, z, c=colors, s=20, alpha=0.7, depthshade=True)
        
        ax.set_title(f'3D Bit Sequence - Polarization {pol_idx}')
        ax.set_xlabel('Time (Sample Index)')
        ax.set_ylabel('Bit Value')
        ax.set_zlabel('Polarization')
        ax.set_yticks([0, 1])
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Bit 1'),
            Patch(facecolor='blue', label='Bit 0')
        ]
        ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('./results/3d_bit_sequences.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_constellation_3d(symbols):
    """在3D空间中可视化星座图"""
    fig = plt.figure(figsize=(15, 12))
    
    for pol_idx in range(symbols.shape[0]):
        pol_symbols = symbols[pol_idx]
        
        # 创建3D子图
        ax = fig.add_subplot(2, 2, pol_idx + 1, projection='3d')
        
        # 取前1000个符号
        n_symbols = min(1000, len(pol_symbols))
        symbols_subset = pol_symbols[:n_symbols]
        
        # 3D坐标：I, Q, 时间
        x = np.real(symbols_subset)  # I分量
        y = np.imag(symbols_subset)  # Q分量
        z = range(n_symbols)  # 时间轴
        
        # 根据幅度着色
        magnitude = np.abs(symbols_subset)
        scatter = ax.scatter(x, y, z, c=magnitude, cmap='viridis', s=20, alpha=0.7, depthshade=True)
        
        ax.set_title(f'3D Constellation - Polarization {pol_idx}\n(Time Evolution)')
        ax.set_xlabel('In-phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.set_zlabel('Time (Symbol Index)')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='Magnitude')
    
    plt.tight_layout()
    plt.savefig('./results/3d_constellation.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_constellation_2d(symbols):
    """传统2D星座图对比"""
    plt.figure(figsize=(12, 5))
    
    for pol_idx in range(symbols.shape[0]):
        pol_symbols = symbols[pol_idx]
        
        plt.subplot(1, symbols.shape[0], pol_idx + 1)
        n_symbols = min(1000, len(pol_symbols))
        
        scatter = plt.scatter(np.real(pol_symbols[:n_symbols]), 
                             np.imag(pol_symbols[:n_symbols]), 
                             c=range(n_symbols), cmap='viridis', s=10, alpha=0.7)
        
        plt.title(f'2D Constellation - Polarization {pol_idx}')
        plt.xlabel('In-phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.colorbar(scatter, label='Time Index')
    
    plt.tight_layout()
    plt.savefig('./results/2d_constellation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_tx_signals_3d(tx_waveforms):
    """在3D空间中可视化发射信号"""
    fig = plt.figure(figsize=(15, 10))
    
    for pol_idx in range(tx_waveforms.shape[0]):
        waveform = tx_waveforms[pol_idx]
        
        # 创建3D子图
        ax = fig.add_subplot(2, 2, pol_idx + 1, projection='3d')
        
        # 取前500个采样点
        n_samples = min(500, len(waveform))
        waveform_subset = waveform[:n_samples]
        
        # 3D坐标：时间，实部，虚部
        x = range(n_samples)  # 时间轴
        y = np.real(waveform_subset)  # 实部
        z = np.imag(waveform_subset)  # 虚部
        
        # 根据相位着色
        phase = np.angle(waveform_subset)
        scatter = ax.scatter(x, y, z, c=phase, cmap='hsv', s=20, alpha=0.7, depthshade=True)
        
        ax.set_title(f'3D Transmit Signal - Polarization {pol_idx}')
        ax.set_xlabel('Time (Sample Index)')
        ax.set_ylabel('Real Part')
        ax.set_zlabel('Imaginary Part')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='Phase (radians)')
    
    plt.tight_layout()
    plt.savefig('./results/3d_tx_signals.png', dpi=150, bbox_inches='tight')
    plt.show()

def explore_output_signals():
    print("\nExplore output signal data...")
    
    file_path = './results/output_signals/output_signals.npz'
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return
    
    data = np.load(file_path)
    
    print("Available data keys:", list(data.keys()))
    
    # 分析原始接收信号
    if 'raw_rx_waveform' in data:
        raw_rx = data['raw_rx_waveform']
        print(f"\nRaw received signal shape: {raw_rx.shape}")
        
        # 3D接收信号可视化
        plot_rx_signals_3d(raw_rx, "Raw_Received")
        
        # 信号质量分析
        analyze_signal_quality_3d(raw_rx)
    
    # 分析处理后的信号
    if 'processed_rx_waveform' in data:
        processed_rx = data['processed_rx_waveform']
        print(f"\nProcessed received signal shape: {processed_rx.shape}")
        
        plot_rx_signals_3d(processed_rx, "Processed_Received")
    
    # 分析解调数据
    if 'demodulated_data' in data:
        demodulated = data['demodulated_data']
        print(f"\nDemodulated data shape: {demodulated.shape}")
        
        plot_demodulated_3d(demodulated)

def plot_rx_signals_3d(rx_waveform, title_prefix):
    """在3D空间中可视化接收信号"""
    fig = plt.figure(figsize=(15, 10))
    
    for pol_idx in range(rx_waveform.shape[0]):
        waveform = rx_waveform[pol_idx]
        
        # 创建3D子图
        ax = fig.add_subplot(2, 2, pol_idx + 1, projection='3d')
        
        # 取前300个采样点
        n_samples = min(300, len(waveform))
        waveform_subset = waveform[:n_samples]
        
        # 3D坐标：时间，实部，虚部
        x = range(n_samples)  # 时间轴
        y = np.real(waveform_subset)  # 实部
        z = np.imag(waveform_subset)  # 虚部
        
        # 根据信噪比估计着色（简化版）
        magnitude = np.abs(waveform_subset)
        noise_estimate = np.std(magnitude)
        snr_colors = magnitude / (noise_estimate + 1e-10)  # 避免除零
        
        scatter = ax.scatter(x, y, z, c=snr_colors, cmap='plasma', s=20, alpha=0.7, depthshade=True)
        
        ax.set_title(f'3D {title_prefix} - Polarization {pol_idx}')
        ax.set_xlabel('Time (Sample Index)')
        ax.set_ylabel('Real Part')
        ax.set_zlabel('Imaginary Part')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='SNR Estimate')
    
    plt.tight_layout()
    plt.savefig(f'./results/3d_{title_prefix.lower()}_signals.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_demodulated_3d(demodulated):
    """在3D空间中可视化解调数据"""
    fig = plt.figure(figsize=(15, 10))
    
    for pol_idx in range(demodulated.shape[0]):
        pol_data = demodulated[pol_idx]
        
        # 创建3D子图
        ax = fig.add_subplot(2, 2, pol_idx + 1, projection='3d')
        
        # 取前500个符号
        n_symbols = min(500, len(pol_data))
        symbols_subset = pol_data[:n_symbols]
        
        # 3D坐标：时间，符号值，出现频率
        x = range(n_symbols)  # 时间轴
        y = symbols_subset  # 符号值 (0-15 for 16QAM)
        
        # 计算每个符号的出现位置（用于z轴）
        unique_symbols, counts = np.unique(symbols_subset, return_counts=True)
        symbol_positions = {sym: idx for idx, sym in enumerate(unique_symbols)}
        z = [symbol_positions[sym] for sym in symbols_subset]
        
        scatter = ax.scatter(x, y, z, c=y, cmap='tab20', s=20, alpha=0.7, depthshade=True)
        
        ax.set_title(f'3D Demodulated Symbols - Polarization {pol_idx}')
        ax.set_xlabel('Time (Symbol Index)')
        ax.set_ylabel('Symbol Value')
        ax.set_zlabel('Symbol Type')
        
        plt.colorbar(scatter, ax=ax, label='Symbol Value')
    
    plt.tight_layout()
    plt.savefig('./results/3d_demodulated_symbols.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_signal_quality_3d(signal):
    """3D信号质量分析"""
    print("\n=== 3D Signal Quality Analysis ===")
    
    for pol_idx in range(signal.shape[0]):
        pol_signal = signal[pol_idx]
        
        amplitude = np.abs(pol_signal)
        phase = np.angle(pol_signal)
        
        print(f"\nPolarization {pol_idx}:")
        print(f"  Amplitude - Mean: {np.mean(amplitude):.4f}, Std: {np.std(amplitude):.4f}")
        print(f"  Phase - Range: {np.rad2deg(np.max(phase)-np.min(phase)):.1f}°")
        print(f"  Dynamic Range: {20*np.log10(np.max(amplitude)/np.min(amplitude)):.1f} dB")

if __name__ == "__main__":
    explore_input_signals()
    explore_output_signals()