import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import os

def upload_parameters():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if not file_path:
        return None
    params = pd.read_excel(file_path, header=None).values
    return params

def generate_save_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx")
    return file_path

def run_simulations(params, data_file):
    n_simulations = int(params[12, 9])  # PM(13,10) em MATLAB é linha 12, coluna 9 em Python (0-index)
    all_data = []
    for j in range(n_simulations):
        # Aqui você deve implementar ou chamar sua função de simulação
        # Exemplo fictício:
        time = np.linspace(0, 48, 100)
        glucose = np.random.rand(100)
        xylose = np.random.rand(100)
        ethanol = np.random.rand(100)
        temperature = np.random.rand(100) * 30 + 20

        df = pd.DataFrame({
            'Time': time,
            'Glucose': glucose,
            'Xylose': xylose,
            'Ethanol': ethanol,
            'Temperature': temperature
        })
        all_data.append(df)

        # Salva cada simulação em uma aba diferente
        with pd.ExcelWriter(data_file, mode='a' if os.path.exists(data_file) else 'w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=f'Sim_{j+1}', index=False)

    messagebox.showinfo("Concluído", f"{n_simulations} simulações salvas em {data_file}")
    return all_data

def main():
    root = tk.Tk()
    root.withdraw()
    params = None
    data_file = None
    all_data = None

    while True:
        action = simpledialog.askinteger(
            "Menu",
            "Escolha uma ação:\n1. Upload parameters\n2. Generate save file\n3. Run simulations\n4. Cancel",
            minvalue=1, maxvalue=4
        )
        if action == 1:
            params = upload_parameters()
            if params is not None:
                messagebox.showinfo("Info", "Parâmetros carregados com sucesso.")
        elif action == 2:
            data_file = generate_save_file()
            if data_file:
                messagebox.showinfo("Info", f"Arquivo para salvar: {data_file}")
        elif action == 3:
            if params is None or data_file is None:
                messagebox.showerror("Erro", "Carregue os parâmetros e gere o arquivo de saída antes de simular.")
            else:
                all_data = run_simulations(params, data_file)
        elif action == 4:
            break

    # Após simulação, perguntar se deseja visualizar
    if all_data:
        view = messagebox.askyesno("Visualizar", "Deseja visualizar os dados simulados?")
        if view:
            import matplotlib.pyplot as plt
            for i, df in enumerate(all_data):
                plt.figure(figsize=(10, 6))
                plt.plot(df['Time'], df['Glucose'], label='Glucose')
                plt.plot(df['Time'], df['Xylose'], label='Xylose')
                plt.plot(df['Time'], df['Ethanol'], label='Ethanol')
                plt.xlabel('Time (h)')
                plt.ylabel('Concentration (g/L)')
                plt.title(f'Simulation {i+1}')
                plt.legend()
                plt.show()

if __name__ == "__main__":
    main()