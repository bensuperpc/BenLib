# ---- Variables ----

# We use variables separate from what CTest uses, because those have
# customization issues
set(
    COVERAGE_GCOVR_COMMAND
    gcovr -r "${PROJECT_SOURCE_DIR}"
    --decisions --calls
    --exclude "${PROJECT_SOURCE_DIR}/external/*"
    --exclude "${PROJECT_SOURCE_DIR}/test/*"
    --exclude "${PROJECT_SOURCE_DIR}/example/*"
    --exclude "${PROJECT_SOURCE_DIR}/build/*"
    --decisions --calls
    --fail-under-line "10"
    --fail-under-branch "10"
    --fail-under-decision "10"
    --fail-under-function "10"
    --html-theme "github.blue"
    --html --html-details --output "${PROJECT_BINARY_DIR}/coverage.html"
    CACHE STRING
    "; separated command to generate a coverage report for the 'coverage' target"
    )

set(
    COVERAGE_TRACE_COMMAND
    lcov --ignore-errors unused,unused --ignore-errors empty,empty
    --ignore-errors inconsistent,inconsistent -c -q
    -o "${PROJECT_BINARY_DIR}/coverage.info"
    -d "${PROJECT_BINARY_DIR}"
    --include "${PROJECT_SOURCE_DIR}/*"
    --exclude "${PROJECT_SOURCE_DIR}/external/*"
    --exclude "${PROJECT_SOURCE_DIR}/test/*"
    --exclude "${PROJECT_SOURCE_DIR}/example/*"
    CACHE STRING
    "; separated command to generate a trace for the 'coverage' target"
)

set(
    COVERAGE_HTML_COMMAND
    genhtml --legend -f -q
    "${PROJECT_BINARY_DIR}/coverage.info"
    -p "${PROJECT_SOURCE_DIR}"
    -o "${PROJECT_BINARY_DIR}/coverage_html"
    CACHE STRING
    "; separated command to generate an HTML report for the 'coverage' target"
)

# ---- Coverage target ----

add_custom_target(
    coverage
    COMMAND ${COVERAGE_GCOVR_COMMAND}
    COMMAND ${COVERAGE_TRACE_COMMAND}
    COMMAND ${COVERAGE_HTML_COMMAND}
    COMMENT "Generating coverage report"
    VERBATIM
)
