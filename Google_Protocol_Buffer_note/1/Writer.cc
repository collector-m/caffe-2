#include "lm.helloworld.pb.h"
#include "iostream"
#include "fstream"
#include <string>
using namespace std;
 int main(void) 
 { 
  
  lm::helloworld msg1; 
  msg1.set_id(101); 
  msg1.set_str("hello"); 
    
  // Write to disk. 
  fstream output("./log", ios::out | ios::trunc | ios::binary); 
        
  if (!msg1.SerializeToOstream(&output)) { 
      cerr << "Failed to write msg." << endl; 
      return -1; 
  }         
	cout<<"Write done \n";

  string str;
  msg1.SerializeToString(&str); // 将对象序列化到字符串，除此外还可以序列化到fstream等

　　printf("%s\n", str.c_str());

  lm::helloworld  x;
　　x.ParseFromString(str); // 从字符串反序列化
　　printf("x.str=%s,s.id = %d \n", x.str().c_str(),x.id()); // 这里的输出将是tom，说明反序列化正确

  return 0; 

 }
