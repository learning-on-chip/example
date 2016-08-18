use sqlite::{Connection, Statement, State};
use std::mem;
use std::path::Path;

use streamer::Result;
use streamer::platform::{self, Platform, Profile};
use streamer::system::{Event, EventKind, Job};

pub struct Database {
    #[allow(dead_code)]
    connection: Connection,
    arrivals: Statement<'static>,
    profiles: Statement<'static>,
}

impl Database {
    pub fn new<T>(platform: &platform::Thermal, path: T) -> Result<Self> where T: AsRef<Path> {
        use sql::prelude::*;

        let connection = ok!(Connection::open(path));

        ok!(connection.execute("
            PRAGMA journal_mode = MEMORY;
            PRAGMA synchronous = OFF;
        "));

        ok!(connection.execute({
            ok!(create_table("arrivals").if_not_exists().columns(&[
                "time".float().not_null(),
            ]).compile())
        }));
        ok!(connection.execute({
            ok!(create_table("profiles").if_not_exists().columns(&[
                "time".float().not_null(), "component_id".integer().not_null(),
                "power".float().not_null(), "temperature".float().not_null(),
            ]).compile())
        }));

        ok!(connection.execute(ok!(delete_from("arrivals").compile())));
        ok!(connection.execute(ok!(delete_from("profiles").compile())));

        let arrivals = {
            let statement = ok!(connection.prepare({
                ok!(insert_into("arrivals").columns(&[
                    "time",
                ]).compile())
            }));
            unsafe { mem::transmute(statement) }
        };
        let profiles = {
            let units = platform.elements().len();
            let statement = ok!(connection.prepare({
                ok!(insert_into("profiles").columns(&[
                    "time", "component_id", "power", "temperature",
                ]).batch(units).compile())
            }));
            unsafe { mem::transmute(statement) }
        };

        Ok(Database { connection: connection, arrivals: arrivals, profiles: profiles })
    }

    pub fn next(&mut self, event: &Event, &(ref power, ref temperature): &(Profile, Profile))
                -> Result<()> {

        ok!(self.connection.execute("BEGIN TRANSACTION"));
        if let &EventKind::Arrived(ref job) = &event.kind {
            ok!(self.write_arrival(job));
        }
        ok!(self.write_profile(power, temperature));
        ok!(self.connection.execute("END TRANSACTION"));
        Ok(())
    }

    fn write_arrival(&mut self, job: &Job) -> Result<()> {
        let statement = &mut self.arrivals;
        ok!(statement.reset());
        ok!(statement.bind(1, job.arrival));
        if State::Done != ok!(statement.next()) {
            raise!("failed to write into the database");
        }
        Ok(())
    }

    fn write_profile(&mut self, power: &Profile, temperature: &Profile) -> Result<()> {
        let &Profile { units, steps, time, time_step, data: ref power } = power;
        let &Profile { data: ref temperature, .. } = temperature;
        let statement = &mut self.profiles;
        for i in 0..steps {
            let time = time + (i as f64) * time_step;
            ok!(statement.reset());
            let mut k = 0;
            for j in 0..units {
                ok!(statement.bind(k + 1, time));
                ok!(statement.bind(k + 2, j as i64));
                ok!(statement.bind(k + 3, power[i * units + j]));
                ok!(statement.bind(k + 4, temperature[i * units + j]));
                k += 4;
            }
            if State::Done != ok!(statement.next()) {
                raise!("failed to write into the database");
            }
        }
        Ok(())
    }
}
