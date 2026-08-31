#!/usr/bin/env bash
# ============================================================================
# run_test_with_bag.sh
# ----------------------------------------------------------------------------
# Test script (static_test.py veya narrow_test.py) ve ros2 bag record'u
# esgudumlu olarak calistirir.
#
# Kullanim:
#   ./run_test_with_bag.sh <ALGORITHM> <SCENARIO>
#
# ALGORITHM : DWA | MPPI | RPP
# SCENARIO  : Static | Narrow_U | Narrow_Z
#             (Dynamic icin ayri wrapper: run_test_with_bag_dynamic.sh)
#
# Cikti:
#   results/<SCENARIO>/<ALGO>_<SCENARIO>_run<N>.csv          (test scripti)
#   results/<SCENARIO>/<ALGO>_<SCENARIO>_run<N>_refpath.csv  (referans yol)
#   results/<SCENARIO>/<ALGO>_<SCENARIO>_run<N>.log          (terminal cikti)
#   bags/<SCENARIO>/<ALGO>_<SCENARIO>_run<N>/                (ros2 bag)
# ============================================================================
set -e
set -u
set -o pipefail

if [ "$#" -ne 2 ]; then
    echo "Kullanim: $0 <ALGORITHM> <SCENARIO>"
    echo "  ALGORITHM: DWA | MPPI | RPP"
    echo "  SCENARIO : Static | Narrow_U | Narrow_Z"
    exit 1
fi

ALGORITHM="$1"
SCENARIO="$2"

# ----------------------------------------------------------------------------
# SCENARIO -> TEST_SCRIPT eslemesi
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(dirname "$SCRIPT_DIR")"

case "$SCENARIO" in
    Static)
        TEST_SCRIPT="$SCRIPT_DIR/static_test.py"
        ;;
    Narrow_U|Narrow_Z)
        TEST_SCRIPT="$SCRIPT_DIR/narrow_corridor_test.py"
        ;;
    Dynamic)
        echo "HATA: Dynamic senaryosu icin ayri wrapper kullanin:"
        echo "      ./run_test_with_bag_dynamic.sh $ALGORITHM"
        exit 1
        ;;
    *)
        echo "HATA: Bilinmeyen SCENARIO: '$SCENARIO'"
        echo "      Gecerli degerler: Static | Narrow_U | Narrow_Z"
        exit 1
        ;;
esac

if [ ! -f "$TEST_SCRIPT" ]; then
    echo "HATA: Test scripti bulunamadi: $TEST_SCRIPT"
    exit 1
fi

# ----------------------------------------------------------------------------
# Yollar
# ----------------------------------------------------------------------------
RESULTS_DIR="$PACKAGE_ROOT/results/$SCENARIO"
BAGS_DIR="$PACKAGE_ROOT/bags/$SCENARIO"

mkdir -p "$RESULTS_DIR"
mkdir -p "$BAGS_DIR"

# Run numarasini on-tespit et
NEXT_RUN=1
while [ -e "$RESULTS_DIR/${ALGORITHM}_${SCENARIO}_run${NEXT_RUN}.csv" ] || \
      [ -e "$BAGS_DIR/${ALGORITHM}_${SCENARIO}_run${NEXT_RUN}" ]; do
    NEXT_RUN=$((NEXT_RUN + 1))
done

BAG_PATH="$BAGS_DIR/${ALGORITHM}_${SCENARIO}_run${NEXT_RUN}"
LOG_PATH="$RESULTS_DIR/${ALGORITHM}_${SCENARIO}_run${NEXT_RUN}.log"

echo "============================================================"
echo "  TEST WRAPPER: ${ALGORITHM} | ${SCENARIO} | run${NEXT_RUN}"
echo "============================================================"
echo "  Script : $(basename "$TEST_SCRIPT")"
echo "  CSV    : $RESULTS_DIR/${ALGORITHM}_${SCENARIO}_run${NEXT_RUN}.csv"
echo "  Log    : $LOG_PATH"
echo "  Bag    : $BAG_PATH"
echo "============================================================"

# ----------------------------------------------------------------------------
# Test scripti icindeki ALGORITHM_NAME/SCENARIO_NAME uyumu (uyari amacli)
# Non-greedy regex (RPP YAML yorumlarindan kaynaklanan greedy bug fix)
# ----------------------------------------------------------------------------
SCRIPT_ALGO=$(grep -E '^ALGORITHM_NAME\s*=' "$TEST_SCRIPT" | head -1 | \
              sed -E 's/^[^"]*"([^"]+)".*/\1/')
SCRIPT_SCEN=$(grep -E '^SCENARIO_NAME\s*=' "$TEST_SCRIPT" | head -1 | \
              sed -E 's/^[^"]*"([^"]+)".*/\1/')

if [ "$SCRIPT_ALGO" != "$ALGORITHM" ] || [ "$SCRIPT_SCEN" != "$SCENARIO" ]; then
    echo "UYARI: $(basename "$TEST_SCRIPT") icindeki:"
    echo "    ALGORITHM_NAME = '$SCRIPT_ALGO'"
    echo "    SCENARIO_NAME  = '$SCRIPT_SCEN'"
    echo "  Komut satiriyla ($ALGORITHM/$SCENARIO) UYUSMUYOR."
    echo "  CSV yanlis isimle kaydedilebilir."
    echo "  Devam: Enter, iptal: Ctrl+C"
    read -r
fi

# ----------------------------------------------------------------------------
# Narrow senaryolarda GOAL_X kontrolu (Narrow_U: -2.5, Narrow_Z: +2.5)
# ----------------------------------------------------------------------------
if [ "$SCENARIO" = "Narrow_U" ] || [ "$SCENARIO" = "Narrow_Z" ]; then
    SCRIPT_GOAL_X=$(grep -E '^GOAL_X\s*,\s*GOAL_Y\s*=' "$TEST_SCRIPT" | head -1 | \
                    sed -E 's/^GOAL_X\s*,\s*GOAL_Y\s*=\s*([+-]?[0-9.]+).*/\1/')
    EXPECTED_GOAL_X=""
    case "$SCENARIO" in
        Narrow_U) EXPECTED_GOAL_X="-2.0" ;;
        Narrow_Z) EXPECTED_GOAL_X="0.0"  ;;
    esac
    if [ -n "$EXPECTED_GOAL_X" ] && [ "$SCRIPT_GOAL_X" != "$EXPECTED_GOAL_X" ] && \
       [ "$SCRIPT_GOAL_X" != "+$EXPECTED_GOAL_X" ]; then
        echo "UYARI: $(basename "$TEST_SCRIPT") icindeki GOAL_X = '$SCRIPT_GOAL_X'"
        echo "       $SCENARIO icin beklenen: '$EXPECTED_GOAL_X'"
        echo "  Devam: Enter, iptal: Ctrl+C"
        read -r
    fi
fi

# ----------------------------------------------------------------------------
# Kayit edilecek topic'ler
# ----------------------------------------------------------------------------
TOPICS=(
    /odom
    /amcl_pose
    /scan
    /cmd_vel
    /plan
    /tf
    /tf_static
    /clock
    /gazebo/model_states
)

echo "ros2 bag record baslatiliyor..."
ros2 bag record "${TOPICS[@]}" -o "$BAG_PATH" \
    > "$BAG_PATH.log" 2>&1 &
BAG_PID=$!

cleanup() {
    if kill -0 "$BAG_PID" 2>/dev/null; then
        echo ""
        echo "Bag kaydi durduruluyor (PID $BAG_PID)..."
        kill -INT "$BAG_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            if ! kill -0 "$BAG_PID" 2>/dev/null; then break; fi
            sleep 1
        done
        if kill -0 "$BAG_PID" 2>/dev/null; then
            echo "Bag SIGINT'e cevap vermedi, SIGTERM gonderiliyor..."
            kill -TERM "$BAG_PID" 2>/dev/null || true
        fi
        wait "$BAG_PID" 2>/dev/null || true
        echo "Bag kaydi kapatildi: $BAG_PATH"
    fi
}
trap cleanup EXIT INT TERM

sleep 2
if ! kill -0 "$BAG_PID" 2>/dev/null; then
    echo "ros2 bag record basarisiz oldu. Log: $BAG_PATH.log"
    cat "$BAG_PATH.log"
    exit 1
fi

echo "Bag kaydi aktif (PID $BAG_PID)"
echo ""

# ----------------------------------------------------------------------------
# Test scriptini calistir
# ----------------------------------------------------------------------------
echo "$(basename "$TEST_SCRIPT") baslatiliyor..."
echo "============================================================"

# tee ile hem ekrana yaz hem dosyaya kaydet
python3 "$TEST_SCRIPT" 2>&1 | tee "$LOG_PATH"

# pipefail acik oldugu icin pipe'in herhangi bir bacagi hata verirse exit kod tasinir
TEST_EXIT_CODE=${PIPESTATUS[0]}

echo "============================================================"
echo "$(basename "$TEST_SCRIPT") tamamlandi (exit code: $TEST_EXIT_CODE)"
echo ""
echo "Run ozeti:"
echo "  CSV : $RESULTS_DIR/${ALGORITHM}_${SCENARIO}_run${NEXT_RUN}.csv"
echo "  Log : $LOG_PATH"
echo "  Bag : $BAG_PATH"

exit $TEST_EXIT_CODE
