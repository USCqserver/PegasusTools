//
// Created by Humberto Munoz Bauza on 5/23/21.
//

#include "pqubit.h"
#include <iostream>

using namespace pgq;
int main(){
    std::cout << "Testing..." << std::endl;

    Pqubit q1(3,   0, 0, 2, 0); // u:0, w:0, k:2, z:0
    
    Pqubit q2 = q1.conn_external(); // u:0, w:0, k:2,  z:1
    Pqubit q3 = q2.conn_internal(-4); // u:1, w:1, k:4, z:0
    Pqubit q4 = q3.conn_internal(3); // u:0 w:0, k:7, z:0
    Pqubit q5 = q4.conn_odd(); //  u:0, w:0, k:6, z:0
    Pqubit q6 = q5.conn_internal(-2); // u:1, w:1, k:2, z:0
    Pqubit q7 = q6.conn_internal(0);  // u:0, w:0, k:8, z:0
    Pqubit q8 = q7.conn_internal(-5); // u:1, w:0, k:7, z:0
    Pqubit q9 = q8.conn_internal(-2);
    std::cout << q1 << std::endl;
    std::cout << q9 << std::endl;
    if(q1 != q9){

        return 1;
    }
    

    return 0;
}