#!/bin/bash

# Configuration
TEST_DIR="build"
LOG_FILE="test_results.log"
GLOBAL_EXIT_CODE=0

# 1. Sanity Check: Does the directory exist?
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Directory '$TEST_DIR' not found."
    echo "Make sure you have built the project before running tests."
    exit 1
fi

echo "Searching for test_* executables in '$TEST_DIR'..."
echo "Output will be captured in $LOG_FILE for analysis."

# Clear previous log
> "$LOG_FILE"

# 2. Enable nullglob: prevents the loop from running once with the literal string 
# "build/test_*" if no files match.
shopt -s nullglob

FOUND_COUNT=0

for test_file in "$TEST_DIR"/test_*; do
    # 3. Check if it is a file and is executable
    if [ -f "$test_file" ] && [ -x "$test_file" ]; then
        ((FOUND_COUNT++))
        echo "---------------------------------------------------"
        echo "Running: $test_file"
        
        # Execute the test and pipe output to both console (tee) and log file
        # We wrap in a subshell to capture exit code correctly with PIPESTATUS if needed,
        # but here we just need to run it.
        "$test_file" | tee -a "$LOG_FILE"
        
        # Check exit code of the test (pipestatus[0] would be the test command)
        TEST_RESULT=${PIPESTATUS[0]}
        
        if [ $TEST_RESULT -ne 0 ]; then
            echo "FAILED: $test_file (Exit code: $TEST_RESULT)"
            GLOBAL_EXIT_CODE=1
        else
            echo "PASSED: $test_file"
        fi
    elif [ -f "$test_file" ]; then
        echo "WARNING: Found '$test_file' but it is not executable. Skipping."
        echo "         Run 'chmod +x $test_file' to fix this."
    fi
done

echo "---------------------------------------------------"

# 4. Final Report
if [ "$FOUND_COUNT" -eq 0 ]; then
    echo "No test executables found in '$TEST_DIR'."
    exit 0 
fi

if [ $GLOBAL_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: All $FOUND_COUNT tests passed."
    
    # 5. Parse logs and generate intermediate stats file
    # We use mktemp to create a safe temporary file for the raw stats
    RAW_STATS=$(mktemp)
    
    awk '
    BEGIN {
        # No header here, we print raw data for sorting
    }
    # Pattern 1: Capture Algorithm Name and Metrics (B/W, MAE, etc.)
    /B\/W=/ {
        name = $1;
        gsub(":", "", name);
        gsub(",", "", $0);

        for(i=1; i<=NF; i++) {
            if($i ~ /^B\/W=/) { split($i, a, "="); bw = a[2]; }
            if($i ~ /^MAE=/)  { split($i, a, "="); mae = a[2]; }
            if($i ~ /^MSE=/)  { split($i, a, "="); mse = a[2]; }
            if($i ~ /^MaxAbs=/) { split($i, a, "="); maxabs = a[2]; }
        }

        sum_bw[name] += bw;
        sum_mae[name] += mae;
        sum_mse[name] += mse;
        sum_maxabs[name] += maxabs;
        count[name]++;
        active_algo = name;
    }

    # Pattern 2: Capture Times
    /CompTime=/ {
        if (active_algo != "") {
            gsub(",", "", $0);
            ct = 0; dt = 0;
            for(i=1; i<=NF; i++) {
                if($i ~ /^CompTime=/)   { split($i, a, "="); ct = a[2]; }
                if($i ~ /^DecompTime=/) { split($i, a, "="); dt = a[2]; }
            }
            sum_comp[active_algo] += ct;
            sum_decomp[active_algo] += dt;
        }
    }

    END {
        # Print Aggregated Results: Name BW Comp Decomp MAE MSE MaxAbs
        for (name in count) {
            n = count[name];
            if (n > 0) {
                printf "%s %.5f %.3f %.3f %.6f %.6f %.6f\n", 
                    name, 
                    sum_bw[name]/n, 
                    sum_comp[name]/n, 
                    sum_decomp[name]/n, 
                    sum_mae[name]/n, 
                    sum_mse[name]/n, 
                    sum_maxabs[name]/n
            }
        }
    }' "$LOG_FILE" > "$RAW_STATS"

    # Define a helper variable for the table formatting awk script to reuse it
    FORMAT_AWK='
    BEGIN {
        print "-------------------------------------------------------------------------------------------------------";
        printf "%-12s | %-10s | %-12s | %-12s | %-10s | %-10s | %-10s\n", "Datatype", "B/W", "Comp(ms)", "Decomp(ms)", "MAE", "MSE", "MaxAbs";
        print "-------------------------------------------------------------------------------------------------------";
    }
    {
        printf "%-12s | %-10.5f | %-12.3f | %-12.3f | %-10.6f | %-10.6f | %-10.6f\n", $1, $2, $3, $4, $5, $6, $7;
    }'

    # --- Table 1: Sorted by B/W (Column 2) ---
    echo ""
    echo "================ PERFORMANCE SUMMARY (Sorted by B/W) ================"
    sort -k2 -n "$RAW_STATS" | awk "$FORMAT_AWK"

    # --- Table 2: Sorted by CompTime (Column 3) ---
    echo ""
    echo "================ PERFORMANCE SUMMARY (Sorted by CompTime) ==========="
    sort -k3 -n "$RAW_STATS" | awk "$FORMAT_AWK"

    # --- Table 3: Sorted by MSE (Column 6) ---
    echo ""
    echo "================ PERFORMANCE SUMMARY (Sorted by MSE) ================"
    sort -k6 -n "$RAW_STATS" | awk "$FORMAT_AWK"

    # Clean up
    rm "$RAW_STATS"

else
    echo "FAILURE: One or more tests failed."
fi

exit $GLOBAL_EXIT_CODE