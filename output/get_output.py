import sys
dir1=sys.argv[1]
dir2=sys.argv[2]

f1=open(dir1,"r")
f2=open(dir2,"w")

lines=f1.readlines()
prev_tag=''
for line in lines:
    line=line.strip()
    tmp=line.split('\t')
    tag=tmp[0]
    sent=tmp[1]
    if tag != prev_tag:
        prev_tag=tag
        f2.write(sent+'\n')

f1.close()
f2.close()