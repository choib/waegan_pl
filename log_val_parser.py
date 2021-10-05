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

    rex1 = re.compile(r"^INFO:root:Namespace\((?:\w+=(?:\w+|\d+|\d+\.\d+|'\w+'),\s)*epoch=(\d+),.*\)$")
    rex2 = re.compile(r"^INFO:root:mean mse:\s+(-?\d+\.\d+(?:e[-+]\d+)?)")
    rex3 = re.compile(r"^INFO:root:File\sA:\['\.\.\/\.\.\/\w+\/\w+\/\w+\/(\d+)\.\w+'\]\s+Test\sLoss:\s(-?\d\.\d+e[+-]\d+)")
    
    with open(rpath,'r') as fp, open(wpath,'w',newline='') as cfp:
        ifp = iter(fp)
        writer = csv.writer(cfp)
        writer.writerow(["Epoch", "total loss", "W File", "mse loss"])
        #for cnt, line in enumerate(fp):
        for line in ifp:
            m = re.findall(rex1, line)
            if m:
                # print(line[(cnt+1) % len(line)])
                # level = list(m)
                # print(level)
                # print(m)
                m0 = m
                # for x in m[0]:
                #    m0.append(x)
                m1 = re.findall(rex2, next(ifp))
                # print(m1)
                if m1:    
                    m0.extend(m1)
                    next(ifp)
                    next(ifp)
                    next(ifp)
                    m2 = re.findall(rex3, next(ifp))
                    if m2:
                        for x in m2[0]:
                            m0.append(x)
                        #m0.extend(m2)
                        writer.writerow(m0)
 

if __name__ == '__main__':
    main()