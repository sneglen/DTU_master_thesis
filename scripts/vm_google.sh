#!/bin/bash

# Example where no VM is running EXTERNAL_IP = <empty>
# NAME              ZONE            MACHINE_TYPE   PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP  STATUS
# vma100-11-us-e1b  us-east1-b      a2-highgpu-1g  true         10.142.0.2                TERMINATED
# vma100-11-eu-w4a  europe-west4-a  a2-highgpu-1g  true         10.164.0.4                TERMINATED

# Example where a VM is running: EXTERNAL_IP = 34.147.37.160
# NAME              ZONE            MACHINE_TYPE   PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP    STATUS
# vma100-11-us-e1b  us-east1-b      a2-highgpu-1g  true         10.142.0.2                  TERMINATED
# vma100-11-eu-w4a  europe-west4-a  a2-highgpu-1g  true         10.164.0.4   34.147.37.160  RUNNING

counter=0

while true; do
    # Try to start the European VM
    ((counter++))
    echo -e "\e[32m ************ Tries: $counter ************ \e[0m"
    gcloud compute instances start vma100-11-eu-w4a --zone europe-west4-a

    # Fetch the external IP; if it is not empty, notify and exit the loop
    EXTERNAL_IP=$(gcloud compute instances list --filter="name=('vma100-11-eu-w4a')" --format="value(EXTERNAL_IP)")
    if [[ -n "$EXTERNAL_IP" ]]; then
        echo -n "$EXTERNAL_IP" | xclip -selection clipboard
        notify-send "VM EUROPE: $EXTERNAL_IP"
        vm_status
        break
    fi

    sleep 60

    # Try to start the American VM
     ((counter++))
    echo -e "\e[32m ************ Tries: $counter ************ \e[0m"
    gcloud compute instances start vma100-11-us-e1b --zone us-east1-b

    # Fetch the external IP; if it is not empty, notify and exit the loop
    EXTERNAL_IP=$(gcloud compute instances list --filter="name=('vma100-11-us-e1b')" --format="value(EXTERNAL_IP)")
    if [[ -n "$EXTERNAL_IP" ]]; then
        echo "$EXTERNAL_IP" | xclip -selection clipboard
        notify-send "VM AMERICA: $EXTERNAL_IP"
        vm_status
        break
    fi

    sleep 60
done
