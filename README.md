# ParallelMatrixMul_OpenCL

## Опис
Проект демонструє порівняння **послідовного множення матриць на CPU** та **паралельного виконання на GPU через OpenCL**.  
Використовується C++ і Intel OpenCL (oneAPI).

---

## Вимоги
- Windows 10  
- Visual Studio 2022 (Community Edition або вище)  
- Intel oneAPI Base Toolkit (включає OpenCL SDK)  
- Драйвер Intel HD Graphics 4600 або сумісний GPU з підтримкою OpenCL  

---

## Інструкція з компіляції та запуску

### 1. **Клонувати репозиторій:**
```bash
git clone https://github.com/MykytaTitarenko/OpenCL-Matrix-Multiplication.git
```
### 2. Відкриття проекту у Visual Studio
1. Створити новий **Console App (C++)**.
2. Додати файл `main.cpp` з репозиторію у проект.
### 3. Підключення OpenCL SDK
1. Відкрити **Project → Properties → C/C++ → General → Additional Include Directories** і додати шлях до заголовків:
```text
C:\Program Files (x86)\Intel\oneAPI\compiler<версія>\windows\include
```
2. Відкрити **Linker → General → Additional Library Directories** і додати шлях до бібліотек:
```text
C:\Program Files (x86)\Intel\oneAPI\opencl<версія>\lib\x64
```
3. В **Linker → Input → Additional Dependencies** додати:
```text
OpenCL.lib
```
### 4. Компіляція та запуск
1. Скомпілювати проект.
2. Запустити програму.  
3. Консоль покаже час виконання на CPU та GPU, а також повідомлення про правильність результату:
```text
Results are correct
```
### 5. Примітки
- Код працюватиме лише на комп’ютерах, де встановлено OpenCL SDK.
- Intel HD Graphics 4600 використовується як GPU для тестування, але можна використовувати інші сумісні GPU.
