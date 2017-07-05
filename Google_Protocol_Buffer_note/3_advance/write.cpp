#include <iostream>
#include <fstream>
#include <string>
#include "addressbook.pb.h"

using namespace std;

void PromptForAddress(tutorial::Persion *persion) {
    cout << "Enter persion name:" << endl;
    string name;
    cin >> name;
    persion->set_name(name);

    int age;
    cin >> age;
    persion->set_age(age);
}

int main(int argc, char **argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " ADDRESS_BOOL_FILE" << endl;
        return -1;
    }

    tutorial::AddressBook address_book;

    {
        fstream input(argv[1], ios::in | ios::binary);
        if (!input) {
            cout << argv[1] << ": File not found. Creating a new file." << endl;
        }
        else if (!address_book.ParseFromIstream(&input)) {
            cerr << "Filed to parse address book." << endl;
            return -1;
        }
    }

    // Add an address
    PromptForAddress(address_book.add_persion());

    {
        fstream output(argv[1], ios::out | ios::trunc | ios::binary);
        if (!address_book.SerializeToOstream(&output)) {
            cerr << "Failed to write address book." << endl;
            return -1;
        }
    }

    // Optional: Delete all global objects allocated by libprotobuf.
    //google::protobuf::ShutdownProtobufLibrary();

    return 0;
}
