# Competency Questions and SPARQL

## CQ-1: Which shipments have severe delays (>= 24h)?
```sparql
SELECT ?shipment ?delayHours
WHERE {
  ?shipment a ex:Shipment ;
            ex:delayHours ?delayHours .
  FILTER (?delayHours >= 24)
}
ORDER BY DESC(?delayHours)
```

## CQ-2: Which carriers are linked to the most SLA violations?
```sparql
SELECT ?carrier (COUNT(?shipment) AS ?violations)
WHERE {
  ?shipment a ex:Shipment ;
            ex:handledBy ?carrier ;
            ex:violates ex:SLA_Standard .
}
GROUP BY ?carrier
ORDER BY DESC(?violations)
```

## CQ-3: Which warehouses are delay hotspots?
```sparql
SELECT ?warehouse (COUNT(?shipment) AS ?delayed)
WHERE {
  ?order a ex:Order ;
         ex:fulfilledFrom ?warehouse ;
         ex:hasShipment ?shipment .
  ?shipment ex:deliveredOnTime ?onTime .
  FILTER (?onTime = false)
}
GROUP BY ?warehouse
ORDER BY DESC(?delayed)
```
