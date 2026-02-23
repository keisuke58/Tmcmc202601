
import os
import re

def indent_lines(lines, spaces=4):
    return [" " * spaces + line if line.strip() else line for line in lines]

def patch_file():
    file_path = "main/case2_main.py"
    with open(file_path, "r") as f:
        content = f.read()
    
    lines = content.splitlines()
    new_lines = []
    
    # Identify blocks using markers
    m1_start_marker = "# ===== STEP 2: M1 TMCMC with Linearization Update ====="
    m1_end_marker = "# ===== STEP 3: M2 TMCMC with Linearization Update ====="
    
    m2_start_marker = "# ===== STEP 3: M2 TMCMC with Linearization Update ====="
    m2_end_marker = "# ===== STEP 4: M3 TMCMC with Linearization Update ====="
    
    m3_start_marker = "# ===== STEP 4: M3 TMCMC with Linearization Update ====="
    m3_end_marker = "# ===== STEP 5: Final Comparison & Summary ====="
    
    # Process line by line
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for M1 start
        if line.strip() == m1_start_marker:
            new_lines.append(line)
            new_lines.append('    if "M1" not in requested_models:')
            new_lines.append('        logger.info("Skipping M1 TMCMC (not requested)")')
            new_lines.append('        samples_M1 = np.zeros((0, 5))')
            new_lines.append('        logL_M1 = np.zeros(0)')
            new_lines.append('        MAP_M1 = np.full(5, np.nan)')
            new_lines.append('        mean_M1 = np.full(5, np.nan)')
            new_lines.append('        converged_M1 = []')
            new_lines.append('        diag_M1 = {"beta_schedules": [], "theta0_history": []}')
            new_lines.append('        time_M1 = 0.0')
            new_lines.append('        map_error_M1 = 0.0')
            new_lines.append('    else:')
            i += 1
            
            # Capture M1 block
            block_lines = []
            while i < len(lines) and lines[i].strip() != m1_end_marker:
                block_lines.append(lines[i])
                i += 1
            
            # Indent M1 block
            new_lines.extend(indent_lines(block_lines))
            continue
            
        # Check for M2 start
        if line.strip() == m2_start_marker:
            new_lines.append(line)
            new_lines.append('    if "M2" not in requested_models:')
            new_lines.append('        logger.info("Skipping M2 TMCMC (not requested)")')
            new_lines.append('        samples_M2 = np.zeros((0, 5))')
            new_lines.append('        logL_M2 = np.zeros(0)')
            new_lines.append('        MAP_M2 = np.full(5, np.nan)')
            new_lines.append('        mean_M2 = np.full(5, np.nan)')
            new_lines.append('        converged_M2 = []')
            new_lines.append('        diag_M2 = {"beta_schedules": [], "theta0_history": []}')
            new_lines.append('        time_M2 = 0.0')
            new_lines.append('        map_error_M2 = 0.0')
            new_lines.append('    else:')
            i += 1
            
            # Capture M2 block
            block_lines = []
            while i < len(lines) and lines[i].strip() != m2_end_marker:
                block_lines.append(lines[i])
                i += 1
            
            # Indent M2 block
            new_lines.extend(indent_lines(block_lines))
            continue

        # Check for M3 start
        if line.strip() == m3_start_marker:
            new_lines.append(line)
            new_lines.append('    if "M3" not in requested_models:')
            new_lines.append('        logger.info("Skipping M3 TMCMC (not requested)")')
            new_lines.append('        samples_M3 = np.zeros((0, 4))')
            new_lines.append('        logL_M3 = np.zeros(0)')
            new_lines.append('        MAP_M3 = np.full(4, np.nan)')
            new_lines.append('        mean_M3 = np.full(4, np.nan)')
            new_lines.append('        converged_M3 = []')
            new_lines.append('        diag_M3 = {"beta_schedules": [], "theta0_history": []}')
            new_lines.append('        time_M3 = 0.0')
            new_lines.append('        map_error_M3 = 0.0')
            new_lines.append('    else:')
            i += 1
            
            # Capture M3 block
            block_lines = []
            while i < len(lines) and lines[i].strip() != m3_end_marker:
                block_lines.append(lines[i])
                i += 1
            
            # Indent M3 block
            new_lines.extend(indent_lines(block_lines))
            continue
            
        new_lines.append(line)
        i += 1
    
    # Add imports or fix indentation if needed (e.g. at EOF)
    
    with open(file_path, "w") as f:
        f.write("\n".join(new_lines))
        
    print("Successfully patched case2_main.py")

if __name__ == "__main__":
    patch_file()
