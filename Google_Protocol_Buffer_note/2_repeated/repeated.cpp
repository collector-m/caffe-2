#include <iostream>
#include "repeated.Person.pb.h"
#include <stdlib.h>
#include <stdio.h>
using namespace std;
/*
message Person {
  required int32 age = 1;
  required string name = 2;
}

message Family {
  repeated Person person = 1;
}
*/
//protobuf repeated类型相当于std的vector，可以用来存放N个相同类型的内容
int main(int argc, char* argv[])
{

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Family family;
    Person* person;

    // 添加一个家庭成员，John
    person = family.add_person();
    person->set_age(25);
    person->set_name("John");

    // 添加一个家庭成员，Lucy
    person = family.add_person();
    person->set_age(23);
    person->set_name("Lucy");

    // 添加一个家庭成员，Tony
    person = family.add_person();
    person->set_age(2);
    person->set_name("Tony");

    // 显示所有家庭成员
    int size = family.person_size();

    cout << "这个家庭有 " << size << " 个成员，如下：" << endl;

    for(int i=0; i<size; i++)
    {
        Person psn = family.person(i);
        cout << i+1 << ". " << psn.name() << ", 年龄 " << psn.age() << endl;
    }

    getchar();
    return 0;
}
