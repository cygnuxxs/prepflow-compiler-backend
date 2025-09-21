from runner import run_code
import textwrap

def test_python():
    code = "print('Hello from Python')\n"
    result = run_code("python", code)
    assert result["success"] and "Hello from Python" in result["output"]
    print("Python test passed!")

def test_c():
    code = textwrap.dedent("""
    #include <stdio.h>
    int main() {
        printf("Hello from C\\n");
        return 0;
    }
    """)
    result = run_code("c", code)
    assert result["success"] and "Hello from C" in result["output"]
    print("C test passed!")

def test_cpp():
    code = textwrap.dedent("""
    #include <iostream>
    int main() {
        std::cout << "Hello from C++\\n";
        return 0;
    }
    """)
    result = run_code("cpp", code)
    assert result["success"] and "Hello from C++" in result["output"]
    print("C++ test passed!")

def test_java():
    code = textwrap.dedent("""
    public class Main {
        public static void main(String[] args) {
            System.out.println("Hello from Java");
        }
    }
    """)
    result = run_code("java", code)
    assert result["success"] and "Hello from Java" in result["output"]
    print("Java test passed!")

def test_javascript():
    code = "console.log('Hello from JS');\n"
    result = run_code("javascript", code)
    assert result["success"] and "Hello from JS" in result["output"]
    print("JavaScript test passed!")

def test_typescript():
    code = "console.log('Hello from TypeScript');\n"
    result = run_code("typescript", code)
    assert result["success"] and "Hello from TypeScript" in result["output"]
    print("TypeScript test passed!")

# New: stdin tests for every language

def test_python_stdin():
    code = "n = int(input())\nprint(n*2)\n"
    result = run_code("python", code, stdin="5")
    assert result["success"], f"Execution failed: {result}"
    assert "10" in result["output"], f"Unexpected output: {result['output']}"
    print("Python stdin test passed!")

def test_c_stdin():
    code = textwrap.dedent("""
    #include <stdio.h>
    int main() {
        int n;
        if (scanf("%d", &n) != 1) return 0;
        printf("%d\\n", n * 2);
        return 0;
    }
    """)
    result = run_code("c", code, stdin="5")
    assert result["success"], f"Execution failed: {result}"
    assert "10" in result["output"], f"Unexpected output: {result['output']}"
    print("C stdin test passed!")

def test_cpp_stdin():
    code = textwrap.dedent("""
    #include <iostream>
    using namespace std;
    int main() {
        int n;
        if (!(cin >> n)) return 0;
        cout << (n * 2) << "\\n";
        return 0;
    }
    """)
    result = run_code("cpp", code, stdin="5")
    assert result["success"], f"Execution failed: {result}"
    assert "10" in result["output"], f"Unexpected output: {result['output']}"
    print("C++ stdin test passed!")

def test_java_stdin():
    code = textwrap.dedent("""
    import java.util.*;
    public class Main {
        public static void main(String[] args) {
            Scanner sc = new Scanner(System.in);
            if (!sc.hasNextInt()) { sc.close(); return; }
            int n = sc.nextInt();
            System.out.println(n * 2);
            sc.close();
        }
    }
    """)
    result = run_code("java", code, stdin="5")
    assert result["success"], f"Execution failed: {result}"
    assert "10" in result["output"], f"Unexpected output: {result['output']}"
    print("Java stdin test passed!")

def test_javascript_stdin():
    code = textwrap.dedent("""
    const fs = require('fs');
    const data = fs.readFileSync(0, 'utf8').trim();
    const n = parseInt(data, 10);
    if (!Number.isNaN(n)) {
      console.log(n * 2);
    }
    """)
    result = run_code("javascript", code, stdin="5")
    assert result["success"], f"Execution failed: {result}"
    assert "10" in result["output"], f"Unexpected output: {result['output']}"
    print("JavaScript stdin test passed!")

def test_typescript_stdin():
    code = textwrap.dedent("""
    declare var require: any;
    const fs = require('fs');
    const data: string = fs.readFileSync(0, 'utf8').trim();
    const n: number = parseInt(data, 10);
    if (!Number.isNaN(n)) {
      console.log(n * 2);
    }
    """)
    result = run_code("typescript", code, stdin="5")
    assert result["success"], f"Execution failed: {result}"
    assert "10" in result["output"], f"Unexpected output: {result['output']}"
    print("TypeScript stdin test passed!")

if __name__ == "__main__":
    test_python()
    test_c()
    test_cpp()
    test_java()
    test_javascript()
    test_typescript()

    # stdin tests
    test_python_stdin()
    test_c_stdin()
    test_cpp_stdin()
    test_java_stdin()
    test_javascript_stdin()
    test_typescript_stdin()
    print("All tests passed with stdin!")
