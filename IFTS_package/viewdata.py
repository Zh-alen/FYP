import numpy as np
import os

def display_binary_signals():
    """显示具体的1和0编码组成的输入和输出信号"""
    
    print("=" * 60)
    print("BINARY SIGNAL ANALYSIS - INPUT AND OUTPUT ENCODING")
    print("=" * 60)
    
    # 检查输入信号文件
    input_file = './results/input_signals/input_signals.npz'
    if not os.path.exists(input_file):
        print(f"ERROR: Input signals file not found: {input_file}")
        # 列出可能的位置
        if os.path.exists('./results/'):
            print("Available files in results directory:")
            for root, dirs, files in os.walk('./results/'):
                for file in files:
                    print(f"  {os.path.join(root, file)}")
        return
    
    # 检查输出信号文件
    output_file = './results/output_signals/output_signals.npz'
    if not os.path.exists(output_file):
        print(f"ERROR: Output signals file not found: {output_file}")
        return
    
    # 加载输入信号数据
    print("\n📥 LOADING INPUT SIGNALS...")
    input_data = np.load(input_file, allow_pickle=True)
    
    # 加载输出信号数据
    print("📤 LOADING OUTPUT SIGNALS...")
    output_data = np.load(output_file, allow_pickle=True)
    
    print("\nINPUT DATA KEYS:", list(input_data.keys()))
    print("OUTPUT DATA KEYS:", list(output_data.keys()))
    
    # 分析输入比特序列
    print("\n" + "=" * 50)
    print("INPUT BINARY ENCODING ANALYSIS")
    print("=" * 50)
    
    if 'bit_sequences' in input_data:
        bits = input_data['bit_sequences']
        print(f"Input bits type: {type(bits)}, dtype: {bits.dtype}, shape: {bits.shape}")
        
        # 处理对象数组
        if bits.dtype == object:
            try:
                bits_list = bits.item() if hasattr(bits, 'item') else bits
                print(f"Total channels: {len(bits_list)}")
                
                for channel_idx, channel_bits in enumerate(bits_list[:2]):  # 只显示前2个信道
                    print(f"\n--- Channel {channel_idx} ---")
                    print(f"Polarizations: {len(channel_bits)}")
                    
                    for pol_idx, pol_bits in enumerate(channel_bits[:2]):  # 只显示前2个偏振
                        print(f"\n  Polarization {pol_idx}:")
                        print(f"    Shape: {pol_bits.shape}")
                        print(f"    Total bits: {len(pol_bits)}")
                        
                        # 显示前50个比特
                        binary_string = ''.join(str(int(bit)) for bit in pol_bits[:50])
                        print(f"    First 50 bits: {binary_string}")
                        
                        # 统计信息
                        zero_count = np.sum(pol_bits == 0)
                        one_count = np.sum(pol_bits == 1)
                        total_bits = len(pol_bits)
                        
                        print(f"    Statistics:")
                        print(f"      Zeros: {zero_count} ({zero_count/total_bits*100:.2f}%)")
                        print(f"      Ones: {one_count} ({one_count/total_bits*100:.2f}%)")
                        
            except Exception as e:
                print(f"Error processing input bits: {e}")
    else:
        print("No 'bit_sequences' found in input data")
        print("Available keys in input data:")
        for key in input_data.keys():
            arr = input_data[key]
            print(f"  {key}: {type(arr)}, {arr.dtype}, {arr.shape if hasattr(arr, 'shape') else 'no shape'}")
    
    # 分析输出信号
    print("\n" + "=" * 50)
    print("OUTPUT SIGNAL ANALYSIS")
    print("=" * 50)
    
    # 分析解调后的数据
    if 'demodulated_data' in output_data:
        demodulated = output_data['demodulated_data']
        print(f"Demodulated output data:")
        print(f"  Shape: {demodulated.shape}")
        print(f"  Data type: {demodulated.dtype}")
        
        # 显示前20个解调值（分别显示两个偏振）
        print(f"  First 20 demodulated values:")
        print(f"    Polarization 0: {demodulated[0, :20]}")
        print(f"    Polarization 1: {demodulated[1, :20]}")
        
        # 统计信息
        print(f"  Statistics for Polarization 0:")
        unique_vals, counts = np.unique(demodulated[0], return_counts=True)
        for val, count in zip(unique_vals, counts):
            print(f"    {val}: {count} occurrences")
    
    # 分析输出比特序列
    print(f"\nOUTPUT BINARY ANALYSIS")
    
    if 'demodulated_data' in output_data:
        demodulated = output_data['demodulated_data']
        
        print("Reconstructed binary from demodulated integers:")
        print(f"Data shape: {demodulated.shape}")
        
        # 处理每个偏振
        for pol_idx in range(demodulated.shape[0]):
            print(f"\n--- Polarization {pol_idx} ---")
            pol_data = demodulated[pol_idx]
            
            # 确定需要的比特数（16QAM需要4比特）
            bits_needed = 4  # 对于16QAM调制
            
            print(f"Bits per symbol: {bits_needed}")
            print(f"Binary representation of first 20 symbols:")
            
            for i in range(min(20, len(pol_data))):
                value = pol_data[i]
                binary_str = bin(int(value))[2:].zfill(bits_needed)  # 明确转换为int
                print(f"  Symbol {i:2d}: {value:2d} = {binary_str}")
    
    # 比较输入和输出
    print("\n" + "=" * 50)
    print("MODULATION SCHEME ANALYSIS")
    print("=" * 50)
    
    print("Based on the output data (values 0-15), this appears to be 16QAM modulation.")
    print("Each symbol represents 4 bits of information.")
    print("\nMapping for 16QAM:")
    print("  0  = 0000     4  = 0100     8  = 1000    12  = 1100")
    print("  1  = 0001     5  = 0101     9  = 1001    13  = 1101") 
    print("  2  = 0010     6  = 0110    10  = 1010    14  = 1110")
    print("  3  = 0011     7  = 0111    11  = 1011    15  = 1111")
    
    # 保存详细的报告
    print("\n" + "=" * 50)
    print("SAVING DETAILED REPORT...")
    
    report_file = './results/binary_analysis_report.txt'
    os.makedirs('./results/', exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("DETAILED BINARY SIGNAL ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODULATION: 16QAM (4 bits per symbol)\n\n")
        
        # 写入输出解调数据
        if 'demodulated_data' in output_data:
            demodulated = output_data['demodulated_data']
            f.write("OUTPUT DEMODULATED SYMBOLS:\n")
            
            for pol_idx in range(demodulated.shape[0]):
                f.write(f"\nPolarization {pol_idx}:\n")
                f.write("Symbol -> Binary mapping (first 100 symbols):\n")
                
                pol_data = demodulated[pol_idx]
                for i in range(min(100, len(pol_data))):
                    value = pol_data[i]
                    binary_str = bin(int(value))[2:].zfill(4)
                    f.write(f"  Pos {i:3d}: {value:2d} = {binary_str}\n")
    
    print(f"Detailed report saved to: {report_file}")
    print("=" * 60)
    print("ANALYSIS COMPLETE!")

if __name__ == "__main__":
    display_binary_signals()