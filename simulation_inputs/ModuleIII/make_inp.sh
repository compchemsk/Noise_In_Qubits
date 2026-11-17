#ls struct_*.xyz | sort -V | sed 's/struct_//; s/\.xyz//' | xargs
#!/bin/bash

for xyz in struct_*.xyz
do
    base=${xyz%.xyz}

cat > ${base}.inp << EOF
! UKS B2PLYP D3BJ EPR-II AUTOAUX TIGHTSCF

%maxcore 4000

%pal
   nprocs 40
end

%scf
   BrokenSym 1,1
end

* xyzfile 0 3 ${xyz}
EOF

done
