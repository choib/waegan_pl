import re
import csv
import sys
import os

def main():
    rpath = sys.argv[1]
    wpath = sys.argv[2]
    if not os.path.isfile(rpath):
       print("File path {} does not exist. Exiting...".format(rpath))
       sys.exit()
    # if not os.path.isfile(wpath):
    #    print("File path {} does not exist. Exiting...".format(rpath))
    #    sys.exit()

    rex1 = re.compile(r"^Epoch:\s\[(\d+)\/\d+\],\[\w+\s+\d+\/\d+\]\s+[D-H]\sLoss:\s?(-?\d\.\d+e[+-]\d{2})\s+[D-H]\sloss:\s?(-?\d\.\d+e[+-]\d{2})\s+[D-H]\sloss:\s?(-?\d\.\d+e[+-]\d{2})\s+[D-H]\sloss:\s?(-?\d\.\d+e[+-]\d{2})\s+[D-H]\sloss:\s?(-?\d\.\d+e[+-]\d{2})(?:\s+gp\sloss:\s?(-?\d\.\d+e[+-]\d{2}))?.*$")
    rex2 = re.compile(r"^INFO:root:\s+Test:\s+(-?\d\.\d+e[-+]\d+)")
    
    with open(rpath,'r') as fp, open(wpath,'w',newline='') as cfp:
        ifp = iter(fp)
        writer = csv.writer(cfp)
        writer.writerow(["Epoch", "G Loss", "E Loss", "D Loss", "F Loss", "H Loss", "gp Loss", "Test Loss"])
        #for cnt, line in enumerate(fp):
        for line in ifp:
            m = re.findall(rex1, line)
            if m:
                #print(line[(cnt+1) % len(line)])
                # level = list(m)
                # print(level)
                m0 = []
                for x in m[0]:
                    m0.append(x)
                m1 = re.findall(rex2, next(ifp))
                if m1:
                    if len(m[0])>6:
                        m0.extend(m1)
                    else:
                        m0.extend("0.0")
                        m0.extend(m1)
                writer.writerow(m0)
 

if __name__ == '__main__':
    main()