const ARRAY_SIZE: usize = 100;
static mut X_MEAS_CORR: [i32; ARRAY_SIZE] = [0; ARRAY_SIZE];
static mut Z_MEAS_CORR: [i32; ARRAY_SIZE] = [0; ARRAY_SIZE];

#[no_mangle]
fn init(){
    // pass
}

#[no_mangle]
fn init_corrections() {
    unsafe {
        X_MEAS_CORR = [0; ARRAY_SIZE];
        Z_MEAS_CORR = [0; ARRAY_SIZE];
    }
}

#[no_mangle]
fn positive_random_integer() -> i32 {
    (fastrand::i32(..) % 4 + 4) % 4
}

#[no_mangle]
fn update_x_correction(meas:i32, qubit:usize) {
    unsafe{
        X_MEAS_CORR[qubit] += meas
    }
}

#[no_mangle]
fn get_x_correction(qubit:usize) -> i32 {
    unsafe {
        return X_MEAS_CORR[qubit] % 2
    }
}

#[no_mangle]
fn update_z_correction(meas:i32, qubit:usize) {
    unsafe{
        Z_MEAS_CORR[qubit] += meas
    }
}

#[no_mangle]
fn get_z_correction(qubit:usize) -> i32 {
    unsafe {
        return Z_MEAS_CORR[qubit] % 2
    }
}