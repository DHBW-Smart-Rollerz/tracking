## Interfaces (Input & Output)

Da wir aktuell `Float32MultiArray` nutzen, ist die **Reihenfolge der Daten im Array strikt einzuhalten**.

### 1. Input (Subscribed Topics)

Der Tracker erwartet Daten vom `object_detection_node`.

* **Topics:**
* `/object_detection/object` (Bewegliche Objekte)
* `/object_detection/sign` (Statische Schilder)


* **Format:** `Float32MultiArray`
* **Daten-Struktur (Flattened Array):**
Pro erkanntem Objekt müssen exakt **6 Werte** im Array stehen:
```text
[ID, BL_x, BL_y, BR_x, BR_y, Score]

```


* `ID`: Class ID (int)
* `BL_x/y`: Bottom-Left Koordinate (mm)
* `BR_x/y`: Bottom-Right Koordinate (mm)
* `Score`: Confidence (0.0 - 1.0)



### 2. Output (Published Topics)

Hier greift der Planner die Daten ab.

* **Topics:**
* `/object_tracking/tracked_objects`
* `/sign_tracking/tracked_signs`


* **Format:** `Float32MultiArray`
* **Daten-Struktur (Flattened Array):**
Pro getracktem Objekt werden **8 Werte** gesendet:
```text
[Track_ID, Class_ID, x, y, vx, vy, Conf, Width]

```


* **Index 0:** `Track_ID` (Stabile ID über die Zeit)
* **Index 1:** `Class_ID` (Objektklasse)
* **Index 2:** `x` (Position in mm, Fahrzeug-koordinaten)
* **Index 3:** `y` (Position in mm, Fahrzeug-koordinaten)
* **Index 4:** `vx` (Geschwindigkeit x in mm/s)
* **Index 5:** `vy` (Geschwindigkeit y in mm/s)
* **Index 6:** `Confidence` (Tracking-Sicherheit, 0.0-1.0)
* **Index 7:** `Width` (Objektbreite in mm)



---

### 3. Bedeutung der IDs

Der Tracker unterscheidet strikt zwischen **dynamischen Objekten** und **statischen Schildern**. Die eingehenden IDs entsprechen den *gemappten IDs* aus der Object Detection Node.

### a. Object Tracker

Diese Objekte werden auf dem Topic `/object_detection/object` erwartet und besitzen eine Bewegungs-Historie.

| Class ID | Bedeutung | Internes Label |
| --- | --- | --- |
| **2** | Auto | `car` |
| **10** | Fußgänger | `pedestrian` |

### b. Sign Tracker

Diese Objekte werden auf dem Topic `/object_detection/sign` erwartet. Sie werden mit Parametern für statische Objekte getrackt (geringeres Rauschen, längere Lebensdauer).

| Class ID | Bedeutung | Internes Label |
| --- | --- | --- |
| **1** | Stoppschild | `stop-sign` |
| **7** | 30er Zone (Anfang) | `thirty-sign` |
| **8** | Ende 30er Zone | `no-thirty-sign` |
| **9** | Zebrastreifen | `crosswalk-sign` |
| **14** | Parken | `parking-sign` |
| **15** | Links abbiegen | `go-left-sign` |
| **16** | Rechts abbiegen | `go-right-sign` |
| **17** | Vorfahrtstraße | `intersection-right-of-way` |
| **18** | Vorfahrt gewähren | `intersaction-grant-sign` |

---

## Parameter & Koordinaten

* **Einheiten:** Alle Positionen und Geschwindigkeiten sind in **Millimetern (mm)**.
* **Koordinatensystem:** Relativ zum Fahrzeug (übernommen aus Detection).
* **Latenz-Kompensation:** Der Tracker berechnet `dt` dynamisch basierend auf den Message-Timestamps, um Lags im Netzwerk auszugleichen.

## Debugging

Um zu prüfen, ob der Tracker korrekt läuft:

```bash
# Output prüfen (sind IDs stabil? ändern sich x/y plausibel?)
ros2 topic echo /object_tracking/tracked_objects

# Prüfen, ob überhaupt Input ankommt
ros2 topic echo /object_detection/object

```