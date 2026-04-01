#!/bin/bash
# Skill 6 Quick Test Script

set -e

WORKSPACE=/workspace
SKILL_DIR=$WORKSPACE/.github/skills/06-comprehensive-review

echo "🧪 Testing Skill 6: Comprehensive Review"
echo "=========================================="
echo ""

# Test 1: Check file existence
echo "✓ Test 1: Checking script files..."
for file in run-review.py checkers/doc_checker.py checkers/skills_checker.py checkers/tools_checker.py checkers/integration_checker.py checkers/report_generator.py; do
    if [ ! -f "$SKILL_DIR/$file" ]; then
        echo "  ❌ Missing: $file"
        exit 1
    fi
    echo "  ✓ Found: $file"
done

# Test 2: Check Python syntax (using grep as basic validation)
echo ""
echo "✓ Test 2: Validating Python syntax patterns..."
for pyfile in $SKILL_DIR/{run-review.py,checkers/*.py}; do
    # Check for basic structure
    if ! grep -q 'def run\|import\|from\|return' "$pyfile"; then
        echo "  ❌ Invalid structure: $(basename $pyfile)"
        exit 1
    fi
    echo "  ✓ Valid structure: $(basename $pyfile)"
done

# Test 3: Count lines of code
echo ""
echo "✓ Test 3: Code statistics..."
TOTAL_LINES=$(wc -l $SKILL_DIR/{run-review.py,checkers/*.py} | tail -1 | awk '{print $1}')
echo "  Total lines: $TOTAL_LINES"

# Test 4: Check README
echo ""
echo "✓ Test 4: Checking README..."
if [ ! -f "$SKILL_DIR/README.md" ]; then
    echo "  ❌ README.md missing"
    exit 1
fi
LINES=$(wc -l < "$SKILL_DIR/README.md")
echo "  README lines: $LINES"

# Test 5: Check directory structure
echo ""
echo "✓ Test 5: Directory structure..."
for dir in checkers config; do
    if [ ! -d "$SKILL_DIR/$dir" ]; then
        echo "  ❌ Missing directory: $dir"
        exit 1
    fi
    echo "  ✓ Directory exists: $dir"
done

echo ""
echo "=========================================="
echo "✅ All tests passed!"
echo ""
echo "Next steps:"
echo "  1. Run: python3 $SKILL_DIR/run-review.py --help"
echo "  2. Execute comprehensive review: python3 $SKILL_DIR/run-review.py --verbose"
echo "  3. Generate report: python3 $SKILL_DIR/run-review.py --save-report"
