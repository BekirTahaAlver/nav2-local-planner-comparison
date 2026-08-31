#!/usr/bin/env bash
# ============================================================================
# run_test_with_bag.sh
# ----------------------------------------------------------------------------
# static_test.py ve ros2 bag record'u esgudumlu olarak calistirir.
#
# Kullanim:
#   ./run_test_with_bag.sh <ALGORITHM> <SCENARIO>
#
# Cikti:
#   results/<SCENARIO>/<ALGO>_<SCENARIO>_run<N>.csv          (static_test.py)
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
    echo "  SCENARIO : Static | Narrow_Corridor | Dynamic"
    exit 1
fi
ALGORITHM="$1"
SCENARIO="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_SCRIPT="$SCRIPT_DIR/static_test.py"
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
echo "  CSV : $RESULTS_DIR/${ALGORITHM}_${SCENARIO}_run${NEXT_RUN}.csv"
echo "  Log : $LOG_PATH"
echo "  Bag : $BAG_PATH"
echo "============================================================"
# static_test.py'deki ALGORITHM_NAME/SCENARIO_NAME uyumu (uyari amacli)
SCRIPT_ALGO=$(grep -E '^ALGORITHM_NAME\s*=' "$TEST_SCRIPT" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
SCRIPT_SCEN=$(grep -E '^SCENARIO_NAME\s*=' "$TEST_SCRIPT" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
if [ "$SCRIPT_ALGO" != "$ALGORITHM" ] || [ "$SCRIPT_SCEN" != "$SCENARIO" ]; then
    echo "UYARI: static_test.py icindeki ALGORITHM_NAME=$SCRIPT_ALGO,"
    echo "    SCENARIO_NAME=$SCRIPT_SCEN. Komut satiriyla ($ALGORITHM/$SCENARIO)"
    echo "    UYUSMUYOR. CSV yanlis isimle kaydedilebilir."
    echo "    Devam: Enter, iptal: Ctrl+C"
    read -r
fi
# Kayit edilecek topic'ler
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
# --- static_test.py'yi calistir, TUM stdout+stderr'i log dosyasina yaz ---
echo "static_test.py baslatiliyor..."
echo "============================================================"
# tee ile hem ekrana yaz hem dosyaya kaydet
python3 "$TEST_SCRIPT" 2>&1 | tee "$LOG_PATH"
# pipefail acik oldugu icin pipe'in herhangi bir bacagi hata verirse exit kod tasinir
TEST_EXIT_CODE=${PIPESTATUS[0]}
echo "============================================================"
echo "static_test.py tamamlandi (exit code: $TEST_EXIT_CODE)"
echo ""
echo "Run ozeti:"
echo "  CSV : $RESULTS_DIR/${ALGORITHM}_${SCENARIO}_run${NEXT_RUN}.csv"
echo "  Log : $LOG_PATH"
echo "  Bag : $BAG_PATH"
exit $TEST_EXIT_CODE
