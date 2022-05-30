# -*- coding: utf-8 -*-
"""
Created on FRI July 13 02:43:02 2018
"""
import numpy as np
import random 

#Ideja e realizimit:
#
class GrafiNeSat:
    emriFile = ""
    matrica = 0
    satFile = 0
    numriVariablave = 0
    numriNgjyrave = 3
    lidhjetVariablave = []
    tempArr1=[]
    formula=[]

    def __init__(self,_matrica,_emriFile="TestFile"):
        self.emriFile = _emriFile+".cnf"
        self.matrica = _matrica
        self.satFile = []
        self.numriNgjyrave = 3
        self.numriVariablave = np.size(_matrica,1)
        #Qdo variabel mund ti kete tri mundesi. Dmth mund te kete ngjyren A B C.
        #A - Kuqe
        #B - Gjelber
        #C - Kalter
        self.tempArr1 = []
        self.formula = []
        
        for i in range(1,self.numriVariablave + 1 ,1):
            for j in range(self.numriNgjyrave):
                self.tempArr1.append(i+2*(i-1)+j) #Zgjidhja eshte bere nga disa kalkulime matematikore 
            self.formula.append(self.tempArr1)
            self.tempArr1=[]
    
        for i in range(self.numriVariablave):
            tempData = self.formula[i]
            self.vendosSePakuNjeNgjyre(tempData)
            self.mosVendosDyNgjyraNeNjeVend(tempData)
            
        
        self.merrLidhjet(self.matrica)
        self.mosVendos()
        
        self.merrSatFile()
        
    #Ne qdo Object duhet te vendoset vetem nje ngjyre.
    
    def vendosSePakuNjeNgjyre(self,pika):
        pika.append(0)
        self.satFile.append(pika)
    
    def mosVendosDyNgjyraNeNjeVend(self,pika):
        self.satFile.append([-pika[0],-pika[1],0])
        self.satFile.append([-pika[0],-pika[2],0])
        self.satFile.append([-pika[1],-pika[2],0])
        
    def merrLidhjet(self,matrica):
        for i in range(np.size(matrica,1)):
            for j in range(np.size(matrica,1)):
                if matrica[i][j] == 1:
                    if matrica[j][i] == 1:
                        if([i,j] not in self.lidhjetVariablave and [j,i] not in self.lidhjetVariablave):
                            self.lidhjetVariablave.append([i,j])
                            
    #Funksioni i cili nuk do te lejoj qe te vendos
    def mosVendos(self):
        for i in range(len(self.lidhjetVariablave)):
            self.mosVendosNgyrenNjejt(self.lidhjetVariablave[i])
            
    def mosVendosNgyrenNjejt(self,_lidhja):
        numri = int(_lidhja[0])
        temp1 = self.formula[numri]
        numri = int(_lidhja[1])
        temp2 = self.formula[numri]
        
        for i in range (len(temp1)):
            self.satFile.append([-temp1[i],-temp2[i],0])       
    
    def merrSatFile(self):
        temp = False
        row = ''
        for i in range(len(self.satFile)):
            for j in range(len(self.satFile[i])):
                if self.satFile[i][0] != 0:
                    temp = True
                    row = row + str(self.satFile[i][j])+" "
                else:
                    temp = False
                    continue
            if temp:
                row = row +"\n"
        self.cnf_lines = row
        self.no_vars = self.numriVariablave*self.numriNgjyrave
        
        # file = open(self.emriFile,"w")
        # file.write(row)
        # file.close()

def get_graph_coloring_clauses(no_ver, no_edge):
    edges = []
    for edge_idx in range(no_edge):
        src = random.randint(1, no_ver)
        dst = random.randint(1, no_ver)
        while (dst == src):
            dst = random.randint(1, no_ver)
        if not ([src, dst] in edges or [dst, src] in edges):
            edges.append([src, dst])
    
    matrix = np.zeros((no_ver, no_ver))
    for edge in edges:
        matrix[edge[0]-1][edge[1]-1] = 1
        matrix[edge[1]-1][edge[0]-1] = 1

    objGrafiNeSat = GrafiNeSat(matrix, 'test.cnf')
    lines = objGrafiNeSat.cnf_lines.split('\n')

    clauses = []
    for line in lines[:-1]:
        ele_list = line.split(' ')
        clause = []
        for ele in ele_list:
            if int(ele) == 0:
                break
            clause.append(int(ele))
        if len(clause) > 0:
            clauses.append(clause)

    return clauses


#Gjetja e zgjidhjes per nje file te caktuar
# matrix = [[0,0,1,1,1],[0,0,1,1,0],[1,1,0,1,0],[1,1,1,0,1],[1,0,0,1,0]]
# emriFile = 'Output.cnf'
# objGrafiNeSat = GrafiNeSat(matrix,emriFile)

if __name__ == '__main__':
    clauses = get_graph_coloring_clauses(3, 7)
    print(clauses)  