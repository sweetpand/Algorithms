
#include <stack>
#include <string>
#include <iostream>

using namespace std;


int operation(string &S, std::stack<int> &nums, string &word, int i){
	int top_element;
    int num1, num2;
	
	if (i == S.size() || S.at(i) == ' '){
		if (word == "DUP"){
			// push the top of stack
			if (nums.size()>0){
				top_element = nums.top();
				nums.push(top_element);
			}
			else{
				return -1;
			}
		}
		else if (word == "POP"){
			if (nums.size()>0){
				nums.pop();
			}
			else{
				return -1;
			}
		}
		else if (word == "+"){
			if (nums.size()>1){
				num1 = nums.top();
				nums.pop();
				num2 = nums.top();
				nums.pop();
				nums.push(num1+num2);
			}
			else{
				return -1;
			}
		}
		else if (word == "-"){
			if (nums.size()>1){
				num1 = nums.top();
				nums.pop();
				num2 = nums.top();
				nums.pop();
				nums.push(num1-num2);
			}
			else{
				return -1;
			}
		}
		else{
			nums.push(stoi(word));
		}
		word = "";
		
	}
	
	else{
		word = word+S.at(i);
		
	}
	
	return 0;
}

int solution(string &S) {
    int stop;
    unsigned i;
    string word = ""; 
    std::stack<int> nums;
	
    for(i=0; i<S.length(); i++){
        stop = operation(S, nums, word, i);     
        if (stop == -1){
            return stop;
        }
    }
    
    stop = operation(S, nums, word, i); 
    if (stop == -1){
            return stop;
    }
    
    return nums.top();
}


int main()
{
    int res;
    string s;
    
    s = "13 DUP 4 POP 5 DUP + DUP + -";
    res = solution(s);
    
    cout << res;
}
