with open('D:/pythonProject/label/spell_1_use_truth.txt', 'r') as f:
    lines = f.readlines()
    new_lines = []
    for line in lines:
        line = line.strip()
        if line == '2' or line == '4':
            new_lines.append('0\n')
        elif line == '3' or line == '5':
            new_lines.append('1\n')
        elif line == '1':
            new_lines.append('2\n')
        else:
            new_lines.append(line+'\n')

with open('D:/pythonProject/label/spell_1_use_truth.txt', 'w') as f:
    f.writelines(new_lines)
