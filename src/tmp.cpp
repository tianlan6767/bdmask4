#include <stdio.h>

#include <iostream>


using namespace std;

class A
{
private:
	int _n;
	int _k;
	char _a;
};

int main()
{
	std::cout << sizeof(A) << std::endl; //8
	return 0;
}