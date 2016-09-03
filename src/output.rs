use sqlite::{Connection, Statement, State};
use std::mem;

use streamer::{Config, Result};
use streamer::platform::{Platform, Profile, Thermal};
use streamer::system::{Event, EventKind};

pub struct Output {
    time: usize,
    element_id: usize,
    batch_size: usize,
    accumulator: (usize, f64, f64),
    connection: Connection,
    events: Statement<'static>,
    profiles: Statement<'static>,
}

impl Output {
    pub fn new(platform: &Thermal, config: &Config) -> Result<Self> {
        use sql::prelude::*;

        let element_id = *some!(config.get::<i64>("element_id"), "an element ID is required");
        let element_id = match element_id {
            value if value >= 0 && value < platform.elements().len() as i64 => value as usize,
            _ => raise!("the element ID is invalid"),
        };
        let batch_size = *config.get::<i64>("batch_size").unwrap_or(&1);
        let batch_size = match batch_size {
            value if value > 0 => value as usize,
            _ => raise!("the batch size is invalid"),
        };
        let connection = ok!(Connection::open(path!(@unchecked config,
                                                    "an output file is required")));
        ok!(connection.execute("
            PRAGMA journal_mode = MEMORY;
            PRAGMA synchronous = OFF;
        "));
        let events = {
            ok!(connection.execute(
                ok!(create_table("events").if_not_exists().columns(&[
                    "time".integer().not_null(),
                    "kind".integer().not_null(),
                ]).compile())
            ));
            ok!(connection.execute(ok!(delete_from("events").compile())));
            let statement = ok!(connection.prepare(
                ok!(insert_into("events").columns(&["time", "kind"]).compile())
            ));
            unsafe { mem::transmute(statement) }
        };
        let profiles = {
            ok!(connection.execute(
                ok!(create_table("profiles").if_not_exists().columns(&[
                    "time".integer().not_null(),
                    "power".float().not_null(),
                    "temperature".float().not_null(),
                ]).compile())
            ));
            ok!(connection.execute(ok!(delete_from("profiles").compile())));
            let statement = ok!(connection.prepare(
                ok!(insert_into("profiles").columns(&["time", "power", "temperature"]).compile())
            ));
            unsafe { mem::transmute(statement) }
        };
        Ok(Output {
            time: 0,
            element_id: element_id,
            batch_size: batch_size,
            accumulator: (0, 0.0, 0.0),
            connection: connection,
            events: events,
            profiles: profiles,
        })
    }

    pub fn next(&mut self, event: &Event, profiles: &(Profile, Profile)) -> Result<()> {
        ok!(self.connection.execute("BEGIN TRANSACTION"));
        ok!(self.process_profiles(profiles));
        ok!(self.process_event(event));
        ok!(self.connection.execute("END TRANSACTION"));
        Ok(())
    }

    fn process_event(&mut self, event: &Event) -> Result<()> {
        let (kind, mapping) = match &event.kind {
            &EventKind::Start(_, ref mapping) => (0, mapping),
            &EventKind::Finish(_, ref mapping) => (1, mapping),
            _ => return Ok(()),
        };
        for &(_, i) in mapping {
            if i == self.element_id {
                try!(self.write_event(kind));
                break;
            }
        }
        Ok(())
    }

    fn process_profiles(&mut self, profiles: &(Profile, Profile)) -> Result<()> {
        let &Profile { element_count, step_count, data: ref power, .. } = &profiles.0;
        let &Profile { data: ref temperature, .. } = &profiles.1;
        for i in 0..step_count {
            self.accumulator.0 += 1;
            self.accumulator.1 += power[i * element_count + self.element_id];
            self.accumulator.2 += temperature[i * element_count + self.element_id];
            if self.accumulator.0 == self.batch_size {
                try!(self.write_profiles());
            }
        }
        Ok(())
    }

    fn write_event(&mut self, kind: i64) -> Result<()> {
        if self.accumulator.0 > 0 {
            try!(self.write_profiles());
        }
        let statement = &mut self.events;
        ok!(statement.reset());
        ok!(statement.bind(1, self.time as i64));
        ok!(statement.bind(2, kind));
        if State::Done != ok!(statement.next()) {
            raise!("failed to write into the database");
        }
        Ok(())
    }

    fn write_profiles(&mut self) -> Result<()> {
        let statement = &mut self.profiles;
        ok!(statement.reset());
        ok!(statement.bind(1, self.time as i64));
        ok!(statement.bind(2, self.accumulator.1 / self.accumulator.0 as f64));
        ok!(statement.bind(3, self.accumulator.2 / self.accumulator.0 as f64));
        if State::Done != ok!(statement.next()) {
            raise!("failed to write into the database");
        }
        self.time += 1;
        self.accumulator = (0, 0.0, 0.0);
        Ok(())
    }
}
