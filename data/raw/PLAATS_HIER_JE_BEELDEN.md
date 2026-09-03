# Ruwe beelden

Zet hier je .bmp/.jpg/.png bestanden neer (submappen mogen), en zet in
`config/default.yaml`:

```yaml
camera:
  source: "folder"
```

De applicatie draait dan op deze beelden in plaats van op de camera. In de HMI
verschijnen dan pijltjesknoppen en "Vastzetten"; de pijltjestoetsen op je
toetsenbord werken ook. Zet `source: "camera"` terug voor de live lijn.
