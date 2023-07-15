with open("dec.txt", 'r', encoding='utf8') as f1:
    with open("bin.txt", 'w', encoding='utf8') as f2:
        for i in f1:
            j = eval(i)
            k = [j[0], j[1]]
            for l in range(2):
                if k[l] >= 0:
                    k[l] = int(bin(k[l])[2:])
                else:
                    k[l] = int(bin(256 + k[l])[2:])
            f2.write(f'({"%08d"%k[0]},{"%08d"%k[1]})\n')
