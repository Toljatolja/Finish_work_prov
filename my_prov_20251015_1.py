import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re 
import numpy as np

# Установим стиль для графиков
plt.style.use('seaborn-v0_8-whitegrid')

# ----------------------------------------------------------------------
# --- ФУНКЦИЯ 1: Чтение данных из файлов INTERMAGNET (IAGA-2002) ---
# ----------------------------------------------------------------------
# Изменено: теперь принимает путь к поддиректории
def read_intermagnet_data(subdir_name='data', file_pattern='spg2022*.min'):
    """
    Читает и объединяет данные из файлов INTERMAGNET в формате IAGA-2002.
    Читает данные из указанной поддиректории (subdir_name) внутри текущей.
    """
    # 1. Формируем полный путь к поддиректории
    current_dir = os.getcwd() # Текущая рабочая директория
    target_dir = os.path.join(current_dir, subdir_name) 
    
    print(f"Поиск файлов в поддиректории: {target_dir}")
    
    # 2. Использование glob для поиска всех файлов по шаблону в целевой директории
    full_path_pattern = os.path.join(target_dir, file_pattern)
    all_files = glob.glob(full_path_pattern)

    if not all_files:
        print(f"⚠️ Ошибка: Файлы по шаблону '{file_pattern}' не найдены в директории '{target_dir}'.")
        return None

    print(f"Найдено файлов: {len(all_files)}")
    
    # --- АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ СТРОКИ ЗАГОЛОВКА ---
    target_filename = all_files[0]
    header_index = -1
    station_code = ''
    
    with open(target_filename, 'r') as f:
        header_lines = [f.readline() for _ in range(30)] 
        
        print("\n--- Анализ служебной информации и поиск заголовка ---")
        for i, line in enumerate(header_lines):
            if 'IAGA Code' in line:
                station_code_match = re.search(r'IAGA Code\s+(\w+)', line)
                station_code = station_code_match.group(1).strip() if station_code_match else ''
                print(f"IAGA Code: {station_code}")
                
            if 'DATE' in line and 'TIME' in line and header_index == -1:
                header_index = i 
                header_line = line.strip()
                print(f"✅ Заголовок найден в строке с индексом {header_index + 1}.")
            
            if any(key in line for key in ['Station Name', 'Reported', 'Data Interval Type', 'Data Type']):
                 print(line.strip())

    if header_index == -1:
        print(f"❌ Критическая ошибка: Не удалось найти строку, содержащую 'DATE' и 'TIME', среди первых 30 строк файла {target_filename}.")
        return None

    column_names = re.split(r'\s+', header_line)
    column_names = [name for name in column_names if name and name not in ['|']]
    
    print(f"Извлеченные заголовки столбцов: {column_names}")
    
    skip_rows_count = header_index + 1 

    list_of_dfs = []

    # Чтение данных: без изменений, использует найденные заголовки
    for filename in all_files:
        try:
            df = pd.read_csv(
                filename,
                skiprows=skip_rows_count,
                header=None,
                names=column_names,
                sep='\s+',
                na_values=['99999.00', '999999.0', '99999'], 
                engine='python'
            )
            
            df = df.dropna(how='all') 
            
            df['DateTime'] = pd.to_datetime(
                df['DATE'].astype(str) + ' ' + df['TIME'].astype(str), 
                format='%Y-%m-%d %H:%M:%S.%f', 
                errors='coerce' 
            )
            df = df.set_index('DateTime')
            
            df = df.drop(columns=['DATE', 'TIME', 'DOY'], errors='ignore')

            new_columns = {}
            for col in df.columns:
                match = re.match(rf'{re.escape(station_code)}([XYZF])', col, re.IGNORECASE)
                if match:
                    new_columns[col] = match.group(1).upper()
                elif col in ['F']: 
                    new_columns[col] = col.upper()
                    
            df = df.rename(columns=new_columns)

            data_columns = [col for col in df.columns if col in ['X', 'Y', 'Z', 'F']]
            df = df[data_columns]
            
            df = df[df.index.notna()]
            
            if not df.empty and len(df.columns) > 0:
                list_of_dfs.append(df)
            else:
                print(f"⚠️ Предупреждение: Не удалось найти геофизические компоненты (X, Y, Z, F) в файле {os.path.basename(filename)}. Файл пропущен.")

        except Exception as e:
            print(f"❌ Критическая Ошибка при чтении файла {os.path.basename(filename)}: {e}")

    if list_of_dfs:
        combined_df = pd.concat(list_of_dfs).sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        print(f"\n✅ Успешно объединены данные. Всего записей: {len(combined_df)}")
        print(f"Временной диапазон: {combined_df.index.min()} - {combined_df.index.max()}")
        return combined_df
    else:
        print("\n❌ Не удалось объединить данные. Список DataFrame пуст.")
        return None

# ----------------------------------------------------------------------
# --- ФУНКЦИЯ 2: Разведочный анализ (EDA) и предобработка данных ---
# (Без изменений)
# ----------------------------------------------------------------------
def preprocess_and_analyze(df):
    """
    Проводит эксплораторный анализ и предобработку геофизических данных.
    """
    if df is None or df.empty:
        print("Невозможно выполнить анализ и предобработку: DataFrame пуст.")
        return None

    # --- 2.1. Эксплораторный анализ (EDA) ---
    print("\n" + "="*50)
    print("--- Эксплораторный Анализ Данных (EDA) ---")
    print("="*50)
    print("\nПервые 5 строк данных (head):")
    print(df.head())
    print("\nОсновные статистики (describe):")
    print(df.describe().T)

    # Анализ пропущенных значений
    print("\nПроцент пропущенных значений (NaNs) до предобработки:")
    nan_percentage = (df.isnull().sum() / len(df)) * 100
    print(nan_percentage[nan_percentage >= 0].sort_values(ascending=False))
    
    # --- 2.2. Предобработка: Заполнение пропусков ---
    print("\n" + "="*50)
    print("--- Предобработка Данных: Линейная Интерполяция ---")
    print("="*50)
    
    df_preprocessed = df.copy()
    initial_nan_count = df_preprocessed.isnull().sum().sum()
    
    df_preprocessed = df_preprocessed.interpolate(method='linear', limit=10) 
    
    final_nan_count = df_preprocessed.isnull().sum().sum()
    print(f"Всего пропусков до интерполяции: {initial_nan_count}")
    print(f"Всего пропусков после интерполяции (limit=10): {final_nan_count}")

    return df_preprocessed

# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# --- ФУНКЦИЯ 3: Построение графиков (С ИЗМЕНЕНИЕМ) ---
# ----------------------------------------------------------------------
def plot_single_component(df, component, title_prefix, preprocessed=False):
    """Строит отдельный график для одной компоненты и СОХРАНЯЕТ ЕГО."""
    if component not in df.columns:
        return

    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df[component], label=component, linewidth=1.0)
        
    status = "(Предобработано)" if preprocessed else "(Исходные данные)"
    title = f"{title_prefix} - Компонента {component} {status}"
    plt.title(title, fontsize=16)
    plt.xlabel("Дата и Время", fontsize=12)
    plt.ylabel(f"Напряженность {component} (нТл)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    
    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: СОХРАНЕНИЕ ГРАФИКА ---
    
    # 1. Создаем имя файла
    filename = f"{component}_{'preproc' if preprocessed else 'raw'}.png"
    
    # 2. Указываем подпапку для сохранения (например, 'plots')
    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir) # Создаем папку, если ее нет
        
    save_path = os.path.join(plot_dir, filename)
    
    # 3. Сохраняем файл
    plt.savefig(save_path)
    plt.close() # Закрываем фигуру, чтобы не перегружать память
    
    print(f"✅ График сохранен: {save_path}")
    
    # Если вы хотите все еще видеть графики в Spyder, добавьте:
    # plt.show() 


def plot_data(df, title_prefix="График Геофизических Данных", preprocessed=False):
    # ... (остальная часть функции без изменений)
    # ...
    components = [col for col in ['X', 'Y', 'Z', 'F'] if col in df.columns]

    for comp in components:
        # Вызываем функцию с сохранением
        plot_single_component(df, comp, title_prefix, preprocessed)

# ... (конец кода)

# ----------------------------------------------------------------------
# --- ОСНОВНАЯ ЧАСТЬ ПРОГРАММЫ ---
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Чтение данных
    print("ЗАПУСК: Чтение данных INTERMAGNET")
    
    # !!! КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Укажите имя поддиректории
    # Если поддиректория называется 'data', используйте 'data'
    # Если поддиректория называется 'min_files', используйте 'min_files'
    SUBDIRECTORY_NAME = 'SPG/2022' 
    
    geodata = read_intermagnet_data(subdir_name=SUBDIRECTORY_NAME, file_pattern='spg2022*.min')

    if geodata is not None:
        # 2. Вывод графиков исходных данных
        plot_data(geodata, title_prefix="ИСХОДНЫЕ Геофизические Данные", preprocessed=False)

        # 3. Разведочный анализ и предобработка
        geodata_processed = preprocess_and_analyze(geodata)

        # 4. Вывод графиков после предобработки
        if geodata_processed is not None:
            plot_data(geodata_processed, title_prefix="Геофизические Данные ПОСЛЕ ПРЕДОБРАБОТКИ", preprocessed=True)
            print("\n✅ Программа завершена. Графики выведены и данные предобработаны.")
    else:
        print("\n❌ Программа завершена. Не удалось обработать данные.")
        print("\n❌ Программа завершена. Не удалось обработать данные.")