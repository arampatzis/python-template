## Documentation

Execute
```
sphinx-apidoc -f -o docs/source/template template/
sphinx-apidoc -f -o docs/source/scripts scripts/
sphinx-build -M html docs/source/ docs/build/
```

and open the html file `docs/build/html/index.html`.
