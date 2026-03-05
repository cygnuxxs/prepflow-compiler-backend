import unittest
from runner import run_code
import textwrap

class TestCodeExecution(unittest.TestCase):
    def test_python(self):
        code = "print('Hello from Python')\n"
        result = run_code("python", code)
        self.assertTrue(result["success"])
        self.assertIn("Hello from Python", result["output"])

    def test_c(self):
        code = textwrap.dedent("""
        #include <stdio.h>
        int main() {
            printf("Hello from C\\n");
            return 0;
        }
        """)
        result = run_code("c", code)
        self.assertTrue(result["success"])
        self.assertIn("Hello from C", result["output"])

    def test_cpp(self):
        code = textwrap.dedent("""
        #include <iostream>
        int main() {
            std::cout << "Hello from C++\\n";
            return 0;
        }
        """)
        result = run_code("cpp", code)
        self.assertTrue(result["success"])
        self.assertIn("Hello from C++", result["output"])

    def test_java(self):
        code = textwrap.dedent("""
        public class Main {
            public static void main(String[] args) {
                System.out.println("Hello from Java");
            }
        }
        """)
        result = run_code("java", code)
        self.assertTrue(result["success"])
        self.assertIn("Hello from Java", result["output"])

    def test_javascript(self):
        code = "console.log('Hello from JS');\n"
        result = run_code("javascript", code)
        self.assertTrue(result["success"])
        self.assertIn("Hello from JS", result["output"])

    def test_typescript(self):
        code = "console.log('Hello from TypeScript');\n"
        result = run_code("typescript", code)
        self.assertTrue(result["success"])
        self.assertIn("Hello from TypeScript", result["output"])

    # Stdin tests
    def test_python_stdin(self):
        code = "n = int(input())\nprint(n*2)\n"
        result = run_code("python", code, stdin="5")
        self.assertTrue(result["success"], f"Execution failed: {result}")
        self.assertIn("10", result["output"], f"Unexpected output: {result['output']}")

    def test_c_stdin(self):
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
        self.assertTrue(result["success"], f"Execution failed: {result}")
        self.assertIn("10", result["output"], f"Unexpected output: {result['output']}")

    def test_cpp_stdin(self):
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
        self.assertTrue(result["success"], f"Execution failed: {result}")
        self.assertIn("10", result["output"], f"Unexpected output: {result['output']}")

    def test_java_stdin(self):
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
        self.assertTrue(result["success"], f"Execution failed: {result}")
        self.assertIn("10", result["output"], f"Unexpected output: {result['output']}")

    def test_javascript_stdin(self):
        code = textwrap.dedent("""
        const fs = require('fs');
        const data = fs.readFileSync(0, 'utf8').trim();
        const n = parseInt(data, 10);
        if (!Number.isNaN(n)) {
          console.log(n * 2);
        }
        """)
        result = run_code("javascript", code, stdin="5")
        self.assertTrue(result["success"], f"Execution failed: {result}")
        self.assertIn("10", result["output"], f"Unexpected output: {result['output']}")

    def test_typescript_stdin(self):
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
        self.assertTrue(result["success"], f"Execution failed: {result}")
        self.assertIn("10", result["output"], f"Unexpected output: {result['output']}")

if __name__ == "__main__":
    unittest.main()
