import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

def import_sbox(file):
    try:
        if file.name.endswith('.xlsx'):
            sbox = pd.read_excel(file, header=None).to_numpy()
        elif file.name.endswith('.csv'):
            sbox = pd.read_csv(file, header=None).to_numpy()
        elif file.name.endswith('.txt'):
            sbox = pd.read_csv(file, header=None, delim_whitespace=True).to_numpy()
        else:
            st.error("Unsupported file format. Please upload .xlsx, .csv, or .txt files.")
            return None

        if sbox.size == 256:  # Element size must be 256
            sbox = sbox.reshape((16, 16))  # Convert into 16x16 matrix
        else:
            st.error("S-Box harus memiliki 256 elemen.")
            return None
        return sbox
    except Exception as e:
        st.error(f"Error importing S-Box: {e}")
        return None

def calculate_nl(sbox):
    non_linearities = []
    for bit in range(8):
        outputs = np.array([((x >> bit) & 1) * 2 - 1 for x in sbox.flatten()])
        spectrum = walsh_hadamard_transform(outputs)
        nl = (256 - np.max(np.abs(spectrum))) // 2
        non_linearities.append(nl)
    return np.mean(non_linearities)

def calculate_sac(sbox):
    sbox = sbox.flatten()
    n = len(sbox)
    total_flips = 0
    total_bits = 0
    for input_val in range(n):
        original_output = binary_representation(sbox[input_val], 8)
        for bit_to_flip in range(8):
            flipped_input = input_val ^ (1 << bit_to_flip)
            flipped_output = binary_representation(sbox[flipped_input], 8)
            bit_flips = sum(1 for orig_bit, flip_bit in zip(original_output, flipped_output) if orig_bit != flip_bit)
            total_flips += bit_flips
            total_bits += 8
    sac_value = total_flips / total_bits
    return sac_value

def calculate_dap(sbox):
    sbox = sbox.flatten()
    n = len(sbox)
    ddt = [[0 for _ in range(n)] for _ in range(n)]
    for x1 in range(n):
        for delta_x in range(n):
            x2 = x1 ^ delta_x
            delta_y = sbox[x1] ^ sbox[x2]
            ddt[delta_x][delta_y] += 1
    max_dap = 0
    for delta_x in range(1, n):
        for delta_y in range(n):
            probability = ddt[delta_x][delta_y] / n
            max_dap = max(max_dap, probability)
    return max_dap

def calculate_lap(sbox):
    sbox = sbox.flatten()
    n = 8
    num_inputs = len(sbox)
    max_lap = 0
    for input_mask in range(1, num_inputs):
        for output_mask in range(1, num_inputs):
            count = 0
            for x in range(num_inputs):
                input_parity = bin(x & input_mask).count('1') % 2
                output_parity = bin(sbox[x] & output_mask).count('1') % 2
                if input_parity == output_parity:
                    count += 1
            lap = abs(count - (num_inputs // 2)) / (num_inputs // 2)
            max_lap = max(max_lap, lap)
    return max_lap / 2

def calculate_bic_nl(sbox):
    non_linearities = []
    for bit in range(8):
        outputs = np.array([((x >> bit) & 1) * 2 - 1 for x in sbox.flatten()])
        spectrum = walsh_hadamard_transform(outputs)
        nl = (256 - np.max(np.abs(spectrum))) // 2
        non_linearities.append(nl)
    return min(non_linearities)

def calculate_bic_sac(sbox):
    n = len(sbox.flatten())
    bit_length = 8
    total_pairs = 0
    total_independence = 0
    for i in range(bit_length):
        for j in range(i + 1, bit_length):
            independence_sum = 0
            for x in range(n):
                for bit_to_flip in range(bit_length):
                    flipped_x = x ^ (1 << bit_to_flip)
                    y1 = sbox.flatten()[x]
                    y2 = sbox.flatten()[flipped_x]
                    independence_sum += ((y1 >> i) & 1 ^ (y2 >> i) & 1) ^ ((y1 >> j) & 1 ^ (y2 >> j) & 1)
            total_independence += independence_sum / (n * bit_length)
            total_pairs += 1
    return total_independence / total_pairs

def walsh_hadamard_transform(f):
    n = len(f)
    hadamard = np.copy(f)
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                x = hadamard[j]
                y = hadamard[j + step]
                hadamard[j] = x + y
                hadamard[j + step] = x - y
        step *= 2
    return hadamard

def binary_representation(num, width):
    return [int(x) for x in f"{num:0{width}b}"]

def export_to_excel(data, filename):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.DataFrame(data).to_excel(writer, index=False, header=False)
    output.seek(0)
    st.download_button(
        label=f"Download {filename}",
        data=output,
        file_name=filename,
        mime="application/vnd.ms-excel"
    )

# Streamlit GUI
st.set_page_config(
    page_title="S-Box Analysis Tool",
    page_icon=":gear:",
    layout="wide"
)

st.title(":gear: **S-Box Analysis Tool**")
st.markdown("The S-Box Analysis Tool is an interactive Streamlit application designed for evaluating the cryptographic security of S-Boxes through metrics such as Nonlinearity (NL), Strict Avalanche Criterion (SAC), Differential Approximation Probability (DAP), and more. This tool offers a user-friendly interface to simplify S-Box analysis, catering to both novice and advanced users.")



input_option = st.radio(
    ":package: **Select S-Box Input Method**",
    ["Upload File", "Manual Input"],
    index=0
)

if input_option == "Upload File":
    uploaded_file = st.file_uploader(":file_folder: Upload your S-Box file", type=['xlsx', 'csv', 'txt'])

    if uploaded_file is not None:
        sbox = import_sbox(uploaded_file)
        if sbox is not None:
            st.write("### :bar_chart: Imported S-Box (Tabel)")
            st.dataframe(pd.DataFrame(sbox, columns=[f"Col {i}" for i in range(1, 17)]))

elif input_option == "Manual Input":
    manual_input = st.text_area(
        ":pencil: **Enter S-Box values** (16x16 matrix):",
        placeholder="Example:\n1,2,3,...,16\n17,18,19,...,32"
    )

    if st.button("Input S-Box"):
        if manual_input:
            try:
                rows = manual_input.strip().split("\n")
                sbox = np.array([list(map(int, row.replace(",", " ").split())) for row in rows])
                if sbox.shape == (16, 16):
                    st.write("### :bar_chart: Manual Input S-Box (Tabel)")
                    st.dataframe(pd.DataFrame(sbox, columns=[f"Col {i}" for i in range(1, 17)]))
                else:
                    st.error(":x: S-Box must be a 16x16 matrix.")
                    sbox = None
            except Exception as e:
                st.error(f":x: Error processing manual input: {e}")
                sbox = None

if 'sbox' in locals() and sbox is not None:
    operation = st.selectbox(":wrench: **Select operation**", ["NL", "SAC", "DAP", "LAP", "BIC-NL", "BIC-SAC"])

    if operation == "NL":
        nl_value = calculate_nl(sbox)
        st.write("### :chart_with_upwards_trend: Nonlinearity (NL) Result")
        st.success(f"**Average NL Value:** {nl_value}")

    elif operation == "SAC":
        sac_value = calculate_sac(sbox)
        st.write("### :chart_with_upwards_trend: Strict Avalanche Criterion (SAC) Result")
        st.success(f"**SAC Value:** {sac_value:.4f}")

    elif operation == "DAP":
        dap_value = calculate_dap(sbox)
        st.write("### :chart_with_upwards_trend: Differential Approximation Probability (DAP) Result")
        st.success(f"**DAP Value:** {dap_value:.4f}")

    elif operation == "LAP":
        lap_value = calculate_lap(sbox)
        st.write("### :chart_with_upwards_trend: Linear Approximation Probability (LAP) Result")
        st.success(f"**LAP Value:** {lap_value:.4f}")

    elif operation == "BIC-NL":
        bic_nl_value = calculate_bic_nl(sbox)
        st.write("### :chart_with_upwards_trend: Bit Independence Criterion - Nonlinearity (BIC-NL) Result")
        st.success(f"**BIC-NL Value:** {bic_nl_value}")

    elif operation == "BIC-SAC":
        bic_sac_value = calculate_bic_sac(sbox)
        st.write("### :chart_with_upwards_trend: Bit Independence Criterion - Strict Avalanche Criterion (BIC-SAC) Result")
        st.success(f"**BIC-SAC Value:** {bic_sac_value:.4f}")
